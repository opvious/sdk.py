from __future__ import annotations

import collections
import dataclasses
import functools
import logging
import pandas as pd
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
)
import weakref

from ...common import Label, to_camel_case
from ..local import LocalSpecification, LocalSpecificationSource
from .identifiers import (
    DefaultIdentifierFormatter,
    GlobalIdentifier,
    Name,
    global_formatting_scope,
)


_logger = logging.getLogger(__name__)


DefinitionCategory = Literal[
    "ALIAS",
    "CONSTRAINT",
    "DIMENSION",
    "OBJECTIVE",
    "PARAMETER",
    "VARIABLE",
]


class Definition:
    """Internal model definition"""

    @property
    def category(self) -> DefinitionCategory:
        raise NotImplementedError()

    @property
    def label(self) -> Optional[Label]:
        raise NotImplementedError()

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        raise NotImplementedError()

    def render_statement(self, label: Label) -> Optional[str]:
        raise NotImplementedError()


class ModelFragment:
    """Model partial"""


def method_decorator(wrapper: Callable[..., Any]) -> Any:
    def wrap(fn):
        return _DecoratedMethod(fn, wrapper)

    return wrap


class _DecoratedMethod:
    def __init__(
        self,
        body: Callable[..., Any],
        wrapper: Callable[[Callable[..., Any]], Any],
    ) -> None:
        self._body = body
        self._wrapper = wrapper
        self._bindings: Any = weakref.WeakKeyDictionary()

    def bound_to(self, owner: Any) -> Any:
        binding = self._bindings.get(owner)
        if not binding:
            binding = self._wrapper(functools.partial(self._body, owner))
            self._bindings[owner] = binding
        return binding

    def __get__(self, owner: Any, _objtype=None) -> Any:
        return self.bound_to(owner)

    def __call__(self, owner) -> Any:  # Property call
        return self.bound_to(owner)()


@dataclasses.dataclass(frozen=True)
class Statement:
    title: str
    category: DefinitionCategory
    label: Label
    name: Optional[Name]
    text: str


@dataclasses.dataclass(frozen=True)
class _Candidate:
    label: Label
    definition: Definition
    model: Model

    def render_statement(self) -> Optional[Statement]:
        d = self.definition
        text = d.render_statement(self.label)
        if text is None:
            return None
        return Statement(
            title=self.model.title,
            category=d.category,
            label=self.label,
            name=d.identifier.format() if d.identifier else None,
            text=text,
        )


@dataclasses.dataclass(frozen=True)
class _Relabeled(ModelFragment):
    fragment: ModelFragment
    labels: Mapping[str, Label]


def relabel(fragment: ModelFragment, **kwargs: Label) -> ModelFragment:
    """Updates a fragment's definitions' labels"""
    return _Relabeled(fragment, kwargs)


class _ModelVisitor:
    def __init__(self) -> None:
        self.candidates: list[_Candidate] = []
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
        owner = fragment or model

        labels: dict[str, Label] = {}
        while isinstance(owner, _Relabeled):
            labels.update(owner.labels)
            owner = owner.fragment

        attrs: dict[str, Any] = {}
        for cls in reversed(owner.__class__.__mro__[1:]):
            attrs.update(cls.__dict__)
        attrs.update(owner.__dict__)
        attrs.update(owner.__class__.__dict__)

        path = [*prefix, ""]
        for attr, value in attrs.items():
            if attr.startswith("_"):
                continue
            path[-1] = attr
            if isinstance(value, property):
                value = value.fget
            if isinstance(value, _DecoratedMethod):
                value = value.bound_to(owner)
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
            self.candidates.append(_Candidate(label, value, model))


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
        return self.__title or self.__class__.__qualname__

    def statements(self) -> Iterable[Statement]:
        visitor = _ModelVisitor()
        visitor.visit(self)
        candidates = visitor.candidates

        by_identifier = {
            c.definition.identifier: c
            for c in candidates
            if c.definition.identifier
        }
        labels_by_identifier = {
            i: d.label for i, d in by_identifier.items() if d.label
        }
        formatter = DefaultIdentifierFormatter(labels_by_identifier)
        reserved = {i.name: i for i in labels_by_identifier if i.name}
        with global_formatting_scope(formatter, reserved):
            idens = set()
            for c in candidates:
                s = c.render_statement()
                if not s:
                    continue
                yield s
                if c.definition.identifier:
                    idens.add(c.definition.identifier)
            for iden in formatter.formatted_globals():
                if iden in idens:
                    continue
                dc = by_identifier.get(iden)
                if not dc:
                    raise Exception(f"Missing candidate: {iden}")
                s = dc.render_statement()
                if not s:
                    raise Exception(f"Missing statement: {iden}")
                yield s

    def definition_counts(self) -> pd.DataFrame:
        df = pd.DataFrame(dataclasses.asdict(s) for s in self.statements())
        grouped: Any = df.groupby(["title", "category"])["text"].count()
        return grouped.unstack(["category"])

    def specification(self) -> LocalSpecification:
        """Generates the model's specification"""
        grouped = collections.defaultdict(list)
        for s in self.statements():
            grouped[s.title].append(s.text)
        joined = {
            title: "".join(f"  {s} \\\\\n" for s in lines)
            for title, lines in grouped.items()
        }
        sources = [
            LocalSpecificationSource(
                title=title,
                text=f"$$\n\\begin{{align*}}\n{contents}\\end{{align*}}\n$$",
            )
            for title, contents in joined.items()
        ]
        return LocalSpecification(sources=sources, description=self.__doc__)
