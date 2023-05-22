from __future__ import annotations

import collections
import dataclasses
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

from ...common import Label, to_camel_case
from ..local import LocalSpecification, LocalSpecificationSource
from .identifiers import (
    DefaultIdentifierFormatter,
    GlobalIdentifier,
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

    def visit(self, model: Model) -> None:
        model_id = id(model)
        if model_id in self._visited:
            return
        self._visited.add(model_id)
        self._visit_fragment(model=model, fragment=None, prefix=model.prefix)
        for dep in model.dependencies:
            self.visit(dep)

    def _visit_fragment(
        self,
        model: Model,
        fragment: Optional[ModelFragment],
        prefix: Sequence[str],
    ) -> None:
        obj = fragment or model

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
            if isinstance(value, ModelFragment):
                self._visit_fragment(model, value, path)
                continue
            if not isinstance(value, Definition):
                continue
            label = (
                labels.get(attr)
                or value.label
                or to_camel_case("_".join(path))
            )
            self.statements.append(_Statement(label, value, fragment, model))


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
            covers = Parameter.indicator(sets, vertices)
            used = Variable.indicator(sets)

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
    __prefix: Optional[Sequence[str]] = None
    __title: Optional[str] = None

    def __init__(
        self,
        dependencies: Optional[Iterable[Model]] = None,
        prefix: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
    ):
        self.__dependencies = list(dependencies) if dependencies else None
        self.__prefix = prefix
        self.__title = title

    @property
    def dependencies(self) -> Sequence[Model]:
        return self.__dependencies or []

    @property
    def prefix(self) -> Sequence[str]:
        return self.__prefix or []

    @property
    def title(self) -> str:
        return self.__title or f"<code>{self.__class__.__name__}</code>"

    def compile_specification_sources(
        self,
    ) -> Sequence[LocalSpecificationSource]:
        """Generates the model's specification sources

        See also :meth:`.Model.compile_specification` for a convenience method
        which also annotates the
        """
        visitor = _ModelVisitor()
        visitor.visit(self)
        statements = visitor.statements

        by_identifier = {
            s.definition.identifier: s
            for s in statements
            if s.definition.identifier
        }
        labels_by_identifier = {
            i: d.label for i, d in by_identifier.items() if d.label
        }
        formatter = DefaultIdentifierFormatter(labels_by_identifier)
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
        return [
            LocalSpecificationSource(
                title=title,
                text=f"$$\n\\begin{{align*}}\n{contents}\\end{{align*}}\n$$",
            )
            for title, contents in contents_by_title.items()
        ]

    async def compile_specification(
        self,
        allow_unused=True,
    ) -> LocalSpecification:
        """Generates the model's specification"""
        sources = self.compile_specification_sources()
        spec = LocalSpecification(sources=sources)
        try:
            codes = ["ERR_UNUSED_DEFINITION"] if allow_unused else []
            spec = await spec.annotated(ignore_codes=codes)
        except Exception:
            _logger.warning("Unable to annotate specification", exc_info=True)
        return spec
