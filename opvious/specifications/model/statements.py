from __future__ import annotations

import collections
import dataclasses
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

from ...common import Label, to_camel_case
from ..local import LocalSpecification, LocalSpecificationSource
from .ast import Space
from .identifiers import (
    Environment,
    GlobalIdentifier,
    IdentifierFormatter,
    Name,
    QuantifierIdentifier,
    global_formatting_scope,
)


_logger = logging.getLogger(__name__)


class Definition:
    """Internal model definition"""

    @property
    def label(self) -> Optional[Label]:
        raise NotImplementedError()

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        raise NotImplementedError()

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        raise NotImplementedError()


class ModelFragment:
    """Model partial"""


@dataclasses.dataclass(frozen=True)
class _Statement:
    label: Label
    definition: Definition
    fragment: Optional[ModelFragment]
    model: Model

    def render(self) -> Optional[str]:
        owner = self.fragment or self.model
        return self.definition.render_statement(self.label, owner)


@dataclasses.dataclass(frozen=True)
class _Relabeled(ModelFragment):
    fragment: ModelFragment
    labels: Mapping[str, Label]


def relabel(fragment: ModelFragment, **kwargs: Label) -> ModelFragment:
    """Updates a fragment's definitions' labels"""
    return _Relabeled(fragment, kwargs)


class _ModelVisitor:
    def __init__(self) -> None:
        self.statements: list[_Statement] = []
        self._visited: set[int] = set()  # Fragment IDs

    def visit(
        self,
        model: Model,
        prefix: Optional[str] = None,
    ) -> None:
        model_id = id(model)
        if model_id in self._visited:
            return
        self._visited.add(model_id)

        self._visit_fragment(model, None, [prefix] if prefix else [])
        for dep in model.dependencies:
            self.visit(dep)

    def _visit_fragment(
        self,
        model: Model,
        frag: Optional[ModelFragment],
        prefix: Sequence[str],
    ) -> None:
        obj = frag or model

        labels: dict[str, Label] = {}
        while isinstance(obj, _Relabeled):
            labels.update(obj.labels)
            obj = obj.fragment

        attrs: dict[str, Any] = {}
        for cls in reversed(obj.__class__.__mro__[1:]):
            attrs.update(cls.__dict__)
        attrs.update(obj.__dict__)
        attrs.update(obj.__class__.__dict__)

        path = [*prefix, ""]
        for attr, value in attrs.items():
            if attr.startswith("_"):
                continue
            path[-1] = attr
            if isinstance(value, property):
                value = value.fget
            if not isinstance(value, Definition):
                if isinstance(value, ModelFragment):
                    self._visit_fragment(model, value, path)
                continue
            label = (
                labels.get(attr)
                or value.label
                or to_camel_case("_".join(path))
            )
            self.statements.append(_Statement(label, value, frag, model))


class Model:
    """An optimization model

    Args:
        dependencies: Optional list of models upon which this model's
            definitions depend. Dependencies' definitions will be automatically
            added when generating this model's specification.

    Toy example for the set cover problem:

    .. code-block:: python

        class SetCover(Model):
            sets = Dimension()
            vertices = Dimension()
            covers = Parameter(sets, vertices, image=indicator())
            used = Variable(sets, image=indicator())

            @constraint
            def all_covered(self):
                for v in self.vertices:
                    count = total(
                        self.used(s) * self.covers(s, v)
                        for s in self.sets
                    )
                    yield count >= 1

            @objective
            def minimize_used(self):
                return total(self.used(s) for s in self.sets)
    """

    __dependencies: Optional[Sequence[Model]] = None
    __prefix: Optional[str] = None
    __title: Optional[str] = None

    def __init__(
        self,
        dependencies: Optional[Iterable[Model]] = None,
        prefix: Optional[Label] = None,
        title: Optional[str] = None,
    ):
        self.__dependencies = list(dependencies) if dependencies else None
        self.__prefix = prefix
        self.__title = title

    @property
    def title(self) -> str:
        return self.__title or f"<code>{self.__class__.__name__}</code>"

    @property
    def dependencies(self) -> Sequence[Model]:
        """The model's dependencies"""
        return self.__dependencies or []

    async def compile_specification(
        self,
        allow_unused=True,
    ) -> LocalSpecification:
        """Generates the model's specification"""
        visitor = _ModelVisitor()
        visitor.visit(self, prefix=self.__prefix)
        statements = visitor.statements

        by_identifier = {
            s.definition.identifier: s
            for s in statements
            if s.definition.identifier
        }
        labels_by_identifier = {
            i: d.label for i, d in by_identifier.items() if d.label
        }
        formatter = _ModelFormatter(labels_by_identifier)
        reserved = {i.name: i for i in labels_by_identifier if i.name}
        rendered_by_title = collections.defaultdict(list)
        with global_formatting_scope(formatter, reserved):
            idens = set()
            for s in statements:
                rs = s.render()
                if not rs:
                    continue
                rendered_by_title[s.model.title].append(rs)
                if s.definition.identifier:
                    idens.add(s.definition.identifier)
            for iden in formatter.formatted_globals():
                if iden in idens:
                    continue
                ds = by_identifier.get(iden)
                if not ds:
                    raise Exception(f"Missing statement: {iden}")
                rs = ds.render()
                if not rs:
                    raise Exception(f"Missing rendered statement: {iden}")
                rendered_by_title[ds.model.title].append(rs)

        contents_by_title = {
            title: "".join(f"  {s} \\\\\n" for s in lines)
            for title, lines in rendered_by_title.items()
        }
        sources = [
            LocalSpecificationSource(
                title=title,
                text=f"$$\n\\begin{{align}}\n{contents}\\end{{align}}\n$$",
            )
            for title, contents in contents_by_title.items()
        ]
        spec = LocalSpecification(sources=sources)

        try:
            codes = ["ERR_UNUSED_DEFINITION"] if allow_unused else []
            spec = await spec.annotated(ignore_codes=codes)
        except Exception:
            _logger.warning("Unable to annotate specification", exc_info=True)
        return spec


class _ModelFormatter(IdentifierFormatter):
    def __init__(self, labels: Mapping[GlobalIdentifier, Label]) -> None:
        super().__init__(labels)

    def _format_dimension(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        if i is None:
            return label[0].upper()
        return f"{label[i]}^\\mathrm{{{label[:i]}}}" if i > 0 else label[i]

    def _format_parameter(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        if not i:
            return label[0].lower()
        return f"{label[i].lower()}^\\mathrm{{{label[:i]}}}"

    def _format_variable(self, label: Label, env: Environment) -> Name:
        i = _last_capital_index(label)
        r = label[i or 0].lower()
        g = _greek_letters.get(r, r)
        if not i:
            return g
        return f"{g}^\\mathrm{{{label[:i]}}}"

    def format_quantifier(
        self, identifier: QuantifierIdentifier, env: Environment
    ) -> Name:
        name = identifier.name
        if not name:
            sp = identifier.space
            if not isinstance(sp, Space):
                raise TypeError(f"Unexpected space: {sp}")
            if hasattr(sp, "identifier") and sp.identifier:  # Dimension
                name = _lower_principal(sp.identifier.format())
            else:  # Interval, possibly aliased
                group = identifier.outer_group
                for g in identifier.groups:
                    if g.rank == 1:
                        group = g
                        break
                if group:
                    name = _lower_principal(group.alias.format())
        return _first_available(name or _DEFAULT_QUANTIFIER_NAME, env)


_DEFAULT_QUANTIFIER_NAME = "x"


def _first_available(name: Name, env: Environment) -> Name:
    while name in env:
        name += "'"
    return name


def _last_capital_index(label: Label) -> Optional[int]:
    j = None
    for i, c in enumerate(label):
        if c.isupper():
            j = i
    return j


def _lower_principal(name: Name) -> Name:
    if "^" not in name:
        return name.lower()
    parts = name.split("^", 1)
    return f"{parts[0].lower()}^{parts[1]}"


_greek_letters = {
    "a": "\\alpha",
    "b": "\\beta",
    "c": "\\chi",
    "d": "\\delta",
    "e": "\\epsilon",
    "f": "\\phi",
    "g": "\\gamma",
    "h": "\\eta",
    "i": "\\iota",
    "j": "\\xi",  # TODO: Find better alternative
    "k": "\\kappa",
    "l": "\\lambda",
    "m": "\\mu",
    "n": "\\nu",
    "o": "\\omicron",
    "p": "\\pi",
    "q": "\\theta",
    "r": "\\rho",
    "s": "\\sigma",
    "t": "\\tau",
    "u": "\\psi",
    "v": "\\zeta",  # TODO: Find better alternative
    "w": "\\omega",
    "x": "\\xi",
    "y": "\\upsilon",
    "z": "\\zeta",
}
