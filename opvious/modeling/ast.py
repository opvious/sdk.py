from __future__ import annotations

import collections
import dataclasses
import itertools
from typing import Any, cast, Iterable, Optional, Sequence, TypeVar, Union

from ..common import encode_extended_float
from .identifiers import (
    Identifier,
    Name,
    local_formatting_scope,
    QuantifierIdentifier,
)
from .lazy import Lazy, force, declare


@dataclasses.dataclass(eq=False, frozen=True)
class _Reference:
    identifier: Identifier
    subscripts: tuple[Expression, ...] = ()

    def render(self, _precedence=0) -> str:
        rendered = self.identifier.format()
        if self.subscripts:
            subscript = ",".join(s.render() for s in self.subscripts)
            rendered += f"_{{{subscript}}}"
        return rendered


# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class Expression:
    def __add__(self, other):
        return _BinaryExpression("add", self, _wrap(other))

    def __radd__(self, left):
        return _literal(left) + self

    def __sub__(self, other):
        return _BinaryExpression("sub", self, _wrap(other))

    def __rsub__(self, left):
        return _literal(left) - self

    def __mod__(self, other):
        return _BinaryExpression("mod", self, _wrap(other))

    def __rmod__(self, left):
        return _literal(left) % self

    def __mul__(self, other):
        return _BinaryExpression("mul", self, _wrap(other))

    def __rmul__(self, left):
        return _literal(left) * self

    def __truediv__(self, other):
        return _BinaryExpression("div", self, _wrap(other))

    def __rtruediv__(self, left):
        return _literal(left) / self

    def __floordiv__(self, other):
        inner = _BinaryExpression("div", self, _wrap(other))
        return _UnaryExpression("floor", inner)

    def __rfloordiv__(self, left):
        return _literal(left) // self

    def __pow__(self, other):
        return _BinaryExpression("pow", self, _wrap(other))

    def __rpow__(self, left):
        return _literal(left) ** self

    def __lt__(self, other):
        return _ComparisonPredicate("<", self, _wrap(other))

    def __le__(self, other):
        return _ComparisonPredicate("\\leq", self, _wrap(other))

    def __eq__(self, other):
        return _ComparisonPredicate("=", self, _wrap(other))

    def __ne__(self, other):
        return _ComparisonPredicate("\\neq", self, _wrap(other))

    def __gt__(self, other):
        return _ComparisonPredicate(">", self, _wrap(other))

    def __ge__(self, other):
        return _ComparisonPredicate("\\geq", self, _wrap(other))

    def __bool__(self):
        return bool(self != 0)

    def render(self, _precedence=0) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(eq=False, frozen=True)
class _LiteralExpression(Expression):
    value: float

    def render(self, _precedence=0) -> str:
        return encode_extended_float(self.value)


def _literal(val) -> _LiteralExpression:
    if not isinstance(val, (float, int)):
        raise TypeError("Unexpected left operand")
    return _LiteralExpression(val)


def _wrap(val) -> Expression:
    if isinstance(val, Expression):
        return val
    return _literal(val)


@dataclasses.dataclass(eq=False, frozen=True)
class ExpressionReference(Expression, _Reference):
    pass


@dataclasses.dataclass(eq=False, frozen=True)
class _UnaryExpression(Expression):
    operator: str
    expression: Expression

    def render(self, _precedence=0) -> str:
        op = self.operator
        return f"\\left\\l{op} {self.expression.render()} \\right\\r{op}"


_binary_operator_precedences = {
    "mul": (4, 4, 4),
    "add": (1, 1, 1),
    "mod": (3, 3, 3),
    "sub": (1, 2, 2),
    "div": (0, 0, 5),
    "pow": (0, 0, 5),
}


@dataclasses.dataclass(eq=False, frozen=True)
class _BinaryExpression(Expression):
    operator: str
    left_expression: Expression
    right_expression: Expression

    def render(self, precedence=0) -> str:
        op = self.operator
        left_inner, right_inner, outer = _binary_operator_precedences[op]
        left = self.left_expression.render(left_inner)
        right = self.right_expression.render(right_inner)
        if op == "mul":
            rendered = f"{left} {right}"
        elif op == "add":
            rendered = f"{left} + {right}"
        elif op == "mod":
            rendered = f"{left} \\bmod {right}"
        elif op == "sub":
            rendered = f"{left} - {right}"
        elif op == "div":
            rendered = f"\\frac{{{left}}}{{{right}}}"
        elif op == "pow":
            rendered = f"\\left({left}\\right)^{{{right}}}"
        else:
            raise Exception(f"Unexpected operator: {op}")
        if outer < precedence:
            rendered = f"\\left({rendered}\\right)"
        return rendered


@dataclasses.dataclass(frozen=True)
class Domain:
    quantifiers: list[QuantifierIdentifier]
    mask: Optional[Predicate]

    def render(self) -> str:
        groups = []
        grouped = itertools.groupby(self.quantifiers, lambda q: q.quantifiable)
        for quantifiable, qs in grouped:
            names = ", ".join(q.format() for q in qs)
            groups.append(f"{names} \\in {quantifiable.render()}")
        rendered = ", ".join(groups)
        if self.mask is not None:
            rendered += f" \\mid {self.mask.render()}"
        return rendered


@dataclasses.dataclass(eq=False, frozen=True)
class _SummationExpression(Expression):
    summand: Expression
    domain: Domain

    def render(self, precedence=0) -> str:
        inner = max(2, precedence)
        with local_formatting_scope(self.domain.quantifiers):
            rendered = f"\\sum_{{{self.domain.render()}}}"
            rendered += f"{{{self.summand.render(inner)}}}"
        return rendered


@dataclasses.dataclass(eq=False, frozen=True)
class _CardinalityExpression(Expression):
    domain: Domain


class Predicate:
    def __and__(self, other):
        return _BinaryPredicate("\\land", self, other)

    def __or__(self, other):
        return _BinaryPredicate("\\lor", self, other)

    def __bool__(self):
        declare(self)
        return True

    def render(self, precedence=0) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _ComparisonPredicate(Predicate):
    command: str
    left_expression: Expression
    right_expression: Expression

    def render(self, _precedence=0) -> str:
        left = self.left_expression.render()
        right = self.right_expression.render()
        return f"{left} {self.command} {right}"


_binary_condition_precedences = {
    "and": 2,
    "or": 1,
}


@dataclasses.dataclass(frozen=True)
class _BinaryPredicate(Predicate):
    condition: str
    left_predicate: Predicate
    right_predicate: Predicate

    def render(self, precedence=0) -> str:
        cond = self.condition
        inner = _binary_operator_precedences[cond]
        left = self.left_predicate.render(inner)
        right = self.right_predicate.render(inner)
        rendered = f"{left} \\l{cond} {right}"
        if inner < precedence:
            rendered = f"\\left({rendered}\\right)"
        return rendered


class Quantifiable:
    def __iter__(self) -> Lazy[Quantifier]:
        return (t[0] for t in cross(self))

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class SpaceReference(_Reference):
    pass


@dataclasses.dataclass(frozen=True)
class Quantifier(Expression):
    identifier: QuantifierIdentifier

    def render(self, _precedence=0) -> str:
        return self.identifier.format()


Space = Lazy[tuple[Quantifier, ...]]


Source = Union[Quantifiable, SpaceReference, Space, tuple["Source", ...]]


_V = TypeVar("_V")


def within_domain(lazy: Lazy[_V]) -> tuple[_V, Domain]:
    value, declarations = force(lazy)
    quantifiers: list[QuantifierIdentifier] = []
    mask: Optional[Predicate] = None
    for declaration in declarations:
        if isinstance(declaration, Predicate):
            if mask:
                mask = _BinaryPredicate("and", mask, declaration)
            else:
                mask = declaration
        elif isinstance(declaration, QuantifierIdentifier):
            quantifiers.append(declaration)
        else:
            raise TypeError(f"Unexpected declaration: {declaration}")
    domain = Domain(quantifiers, mask)
    return value, domain


def domain_from_space(space: Space) -> Domain:
    qs, domain = within_domain(space)
    idens = [q.identifier for q in qs]
    if not _isomorphic(idens, domain.quantifiers):
        raise Exception("Inconsistent quantifiers")
    return dataclasses.replace(domain, quantifiers=idens)


def _isomorphic(
    qs1: Iterable[QuantifierIdentifier], qs2: Iterable[QuantifierIdentifier]
) -> bool:
    return collections.Counter(qs1) == collections.Counter(qs2)


def cross(*sources: Source, names: Optional[Sequence[Name]] = None) -> Space:
    """Generates a cross-product space

    Args:
        sources: One or more sources
        names: Optional name prefixes
    """
    names_by_index = dict(enumerate(names or []))
    yield tuple(
        Quantifier(declare(q.child(name=names_by_index.get(i))))
        for i, q in enumerate(_source_quantifiers(sources))
    )


def _source_quantifiers(source: Source) -> Iterable[QuantifierIdentifier]:
    if isinstance(source, tuple):
        for component in source:
            yield from _source_quantifiers(component)
    elif isinstance(source, (Quantifiable, SpaceReference)):
        yield QuantifierIdentifier.root(source)
    else:  # Space
        domain = domain_from_space(cast(Space, source))
        declare(domain.mask)
        yield from domain.quantifiers


def total(body: Lazy[Expression]) -> Expression:
    return _SummationExpression(*within_domain(body))


def size(space: Space) -> Expression:
    _, domain = within_domain(None for _t in space)
    return _CardinalityExpression(domain)
