from __future__ import annotations

import collections
import dataclasses
import logging
import os
import pandas as pd
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Union

from ..common import Bindable, Label, to_camel_case
from ..specifications.local import LocalSpecification, LocalSpecificationSource
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
    """Base model definition

    Definition instances embedded within a model are automatically detected and
    added to its specification. There are several different types of
    definitions:

    * :class:`dimensions <Dimension>`, input key sets
    * :class:`parameters <Parameter>`, input tensors
    * :class:`variables <Variable>`, output tensors
    * :class:`constraints <Constraint>`, linear constraints
    * :class:`objectives <Objective>`, linear or quadratic targets
    * :class:`aliases <Alias>`, convenience shorthands

    """

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
    """Reusable model sub-component

    Model fragments are useful to group related definitions together and expose
    them in a reusable way. See :ref:`the API reference
    <\\`opvious.modeling.fragments\\`>` for the list of available fragments.
    """

    @property
    def default_definition(self) -> Optional[Label]:
        """Optional label that will be used as the fragment's default.

        The matching definition's final label will be shortened to the
        fragment's prefix.
        """
        return None


@dataclasses.dataclass(frozen=True)
class Statement:
    """A rendered definition"""

    title: str
    """The title of the model the definition belongs to"""

    category: DefinitionCategory
    """The type of definition this statement contains"""

    label: Label
    """The definition's label"""

    name: Optional[Name]
    """The definition's name, if applicable"""

    text: str
    """The definition's mathematical representation"""


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
    """Updates a fragment's definitions' labels

    Args:
        fragment: The fragment containing definitions to relabel
        kwargs: Dictionary from old label to new label
    """
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
        self._visit_owner(model=model, fragment=None, prefix=model.prefix)
        for dep in model.dependencies:
            self.visit(dep)

    def _visit_owner(
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
            if fragment and attr == fragment.default_definition:
                path[-1] = ""
            else:
                path[-1] = attr
            unwrapped = _unwrap_value(value, owner)
            if isinstance(unwrapped, ModelFragment):
                self._visit_owner(model, unwrapped, path)
            elif isinstance(unwrapped, Definition):
                label = (
                    labels.get(attr)
                    or unwrapped.label
                    or to_camel_case("_".join(path))
                )
                self.candidates.append(_Candidate(label, unwrapped, model))
            else:
                _logger.debug(
                    "Ignoring attribute. [path=%s, value=%s]", path, value
                )


def _unwrap_value(
    value: Any, owner: Any
) -> Union[None, ModelFragment, Definition]:
    is_property = isinstance(value, property)
    if is_property:
        value = value.fget
    is_bindable = isinstance(value, Bindable)
    if is_bindable:
        value = value.bound_to(owner)
    if isinstance(value, (ModelFragment, Definition)):
        return value
    if callable(value):
        if is_bindable:
            value = value()
        elif is_property:
            value = value(owner)
    return value if isinstance(value, (ModelFragment, Definition)) else None


class Model:
    """An optimization model

    Args:
        dependencies: Optional list of models upon which this model's
            definitions depend. Dependencies' definitions will be automatically
            added when generating this model's specification.
        prefix: Optional prefix added to all generated labels in this model
        title: Optional title used when creating the model's
            :class:`.LocalSpecification`
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
        """The model's list of dependencies"""
        return self.__dependencies or []

    @property
    def prefix(self) -> Sequence[str]:
        """The model's label prefix, or an empty list if unset"""
        return self.__prefix or []

    @property
    def title(self) -> str:
        """The model's title, defaulting to its class' `__qualname__`"""
        return self.__title or self.__class__.__qualname__

    def statements(self) -> Iterable[Statement]:
        """Lists the model's (and any dependencies') statements"""
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
        """Returns a dataframe summarizing the number of definitions"""
        df = pd.DataFrame(dataclasses.asdict(s) for s in self.statements())
        grouped: Any = df.groupby(["title", "category"])["text"].count()
        return grouped.unstack(["category"]).fillna(0).astype(int)

    def specification(
        self, align: Optional[bool] = None
    ) -> LocalSpecification:
        """Generates the model's specification

        This specification can be used to interact with :class:`.Client`
        methods, for example to start a solve.

        Args:
            align: Embed generated statement definitions inside an `align`
                environment. This can be disabled to enable pretty printing in
                environments which do not support them (e.g. Colab).
        """
        if align is None:
            align = not os.getenv("COLAB_RELEASE_TAG")
        grouped = collections.defaultdict(list)
        for s in self.statements():
            grouped[s.title].append(s.text)
        joined = {
            title: "".join(f"  {s} \\\\\n" for s in lines)
            for title, lines in grouped.items()
        }
        sources = [
            LocalSpecificationSource(
                title=title, text=_format_contents(contents, align)
            )
            for title, contents in joined.items()
        ]
        return LocalSpecification(sources=sources, description=self.__doc__)


def _format_contents(contents, align):
    if align:
        return f"$$\n\\begin{{align*}}\n{contents}\\end{{align*}}\n$$"
    return f"$$\n{contents.replace('&', '')}$$"
