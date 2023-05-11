import dataclasses
from typing import Optional, Sequence

from ..common import Label, to_camel_case
from .definitions import Definition
from .identifiers import (
    AliasIdentifier,
    DimensionIdentifier,
    Environment,
    IdentifierFormatter,
    Name,
    QuantifierIdentifier,
    TensorIdentifier,
    global_formatting_scope,
)


class _Meta(type):
    def __new__(cls, name, bases, dct):
        instance = super().__new__(cls, name, bases, dct)
        statements = dct["_statements"]
        for attr, value in dct.items():
            if not isinstance(value, Definition):
                continue
            label = value.label or to_camel_case(attr)
            statements.append(_Statement(label, value))
        return instance


@dataclasses.dataclass(frozen=True)
class _Statement:
    label: Label
    definition: Definition


class Model(metaclass=_Meta):
    def __init__(
        self, formatter: Optional[IdentifierFormatter] = None
    ) -> None:
        self._formatter = formatter or _DefaultFormatter()
        self._statements: list[_Statement] = []

    def render_specification_source(self) -> str:  # TODO: labels filter
        identifiables = (
            s.definition.identifier
            for s in self._statements
            if s.definition.identifier
        )
        reserved = {i.name: i for i in identifiables if i.name}
        with global_formatting_scope(self._formatter, reserved):
            statements = (
                s.definition.render_statement(s.label, self)
                for s in self._statements
            )
            contents = "".join(s for s in statements if s)
        return f"$$\n\\begin{{align}}\n{contents}\\end{{align}}\n$$"

    def _repr_latex_(self) -> str:
        return self.render_specification_source()


class _DefaultFormatter(IdentifierFormatter):
    def _format_dimension(
        self, dim: DimensionIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()

    def _format_tensor(
        self, tensor: TensorIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()

    def _format_alias(self, alias: AliasIdentifier, env: Environment) -> Name:
        raise NotImplementedError()

    def format_quantifier(
        self, quant: QuantifierIdentifier, env: Environment
    ) -> Name:
        raise NotImplementedError()
