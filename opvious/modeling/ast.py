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
from .quantified import Quantified, unquantify, declare


@dataclasses.dataclass(eq=False, frozen=True)
class _Reference:
    identifier: Identifier
    subscripts: tuple[Expression, ...] = ()


def render_identifier(iden: Identifier, *subscripts: Expression) -> str:
    s = iden.format()
    if subscripts:
        sub = ",".join(s.render() for s in subscripts)
        s += f"_{{{sub}}}"
    return s


# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class Expression:
    def __add__(self, other):
        return _BinaryExpression("add", self, to_expression(other))

    def __radd__(self, left):
        return literal(left) + self

    def __sub__(self, other):
        return _BinaryExpression("sub", self, to_expression(other))

    def __rsub__(self, left):
        return literal(left) - self

    def __mod__(self, other):
        return _BinaryExpression("mod", self, to_expression(other))

    def __rmod__(self, left):
        return literal(left) % self

    def __mul__(self, other):
        return _BinaryExpression("mul", self, to_expression(other))

    def __rmul__(self, left):
        return literal(left) * self

    def __truediv__(self, other):
        return _BinaryExpression("div", self, to_expression(other))

    def __rtruediv__(self, left):
        return literal(left) / self

    def __floordiv__(self, other):
        inner = _BinaryExpression("div", self, to_expression(other))
        return _UnaryExpression("floor", inner)

    def __rfloordiv__(self, left):
        return literal(left) // self

    def __pow__(self, other):
        return _BinaryExpression("pow", self, to_expression(other))

    def __rpow__(self, left):
        return literal(left) ** self

    def __lt__(self, other):
        return _ComparisonPredicate("<", self, to_expression(other))

    def __le__(self, other):
        return _ComparisonPredicate("\\leq", self, to_expression(other))

    def __eq__(self, other):
        return _ComparisonPredicate("=", self, to_expression(other))

    def __ne__(self, other):
        return _ComparisonPredicate("\\neq", self, to_expression(other))

    def __gt__(self, other):
        return _ComparisonPredicate(">", self, to_expression(other))

    def __ge__(self, other):
        return _ComparisonPredicate("\\geq", self, to_expression(other))

    def __bool__(self):
        return bool(self != 0)

    def render(self, _precedence=0) -> str:
        raise NotImplementedError()


ExpressionLike = Union[Expression, float, int]


@dataclasses.dataclass(eq=False, frozen=True)
class _LiteralExpression(Expression):
    value: float

    def render(self, _precedence=0) -> str:
        return str(encode_extended_float(self.value))


def literal(val: Union[float, int]) -> Expression:
    if not isinstance(val, (float, int)):
        raise TypeError("Unexpected literal value")
    return _LiteralExpression(val)


def is_literal(expr: Expression, val: Union[float, int]) -> bool:
    return isinstance(expr, _LiteralExpression) and expr.value == val


def to_expression(val: ExpressionLike) -> Expression:
    if isinstance(val, Expression):
        return val
    if isinstance(val, (float, int)):
        return literal(val)
    if hasattr(val, "to_expression"):
        return val.to_expression()
    raise TypeError(f"Unexpected expression: {val}")


@dataclasses.dataclass(eq=False, frozen=True)
class ExpressionReference(Expression, _Reference):
    def render(self, _precedence=0) -> str:
        return render_identifier(self.identifier, *self.subscripts)


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
    mask: Optional[Predicate] = None

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
        inner = max(3, precedence)
        with local_formatting_scope(self.domain.quantifiers):
            rendered = f"\\sum_{{{self.domain.render()}}}"
            rendered += self.summand.render(inner)
        return rendered


@dataclasses.dataclass(eq=False, frozen=True)
class _CardinalityExpression(Expression):
    domain: Domain

    def render(self, precedence=0) -> str:
        raise NotImplementedError()  # TODO


@dataclasses.dataclass(frozen=True)
class _SwitchCase:
    expression: Expression
    predicate: Optional[Predicate] = None


@dataclasses.dataclass(eq=False, frozen=True)
class _SwitchExpression(Expression):
    cases: Sequence[_SwitchCase]

    def render(self, precedence=0) -> str:
        cs: list[str] = []
        for c in self.cases:
            s = c.expression.render()
            if c.predicate is not None:
                s += f" \\mid {c.predicate.render()}"
            cs.append(s)
        sep = ", \\\\ "
        return f"\\begin{{cases}} {sep.join(cs)} \\end{{cases}}"


class ScalarQuantifiable:
    def __iter__(self) -> Quantified[Quantifier]:
        return (t[0] for t in cross(self))

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class QuantifiableReference(ScalarQuantifiable, _Reference):
    def render(self) -> str:
        return render_identifier(self.identifier, *self.subscripts)


@dataclasses.dataclass(frozen=True)
class Quantifier(Expression):
    identifier: QuantifierIdentifier

    def render(self, _precedence=0) -> str:
        return self.identifier.format()


Quantification = Quantified[tuple[Quantifier, ...]]


def expression_quantifiable(expr: Expression) -> Optional[ScalarQuantifiable]:
    if isinstance(expr, Quantifier):
        return expr.identifier.quantifiable
    return None


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


Quantifiable = Union[
    Quantification,
    Quantified[Quantifier],
    ScalarQuantifiable,
    Domain,
    tuple["Quantifiable", ...],
]


_V = TypeVar("_V")


def within_domain(quantified: Quantified[_V]) -> tuple[_V, Domain]:
    value, declarations = unquantify(quantified)
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


def domain_from_quantifiable(
    quantifiable: Quantifiable, names: Optional[Iterable[Name]] = None
) -> Domain:
    return _domain_from_quantified(cross(quantifiable, names=names))


def _domain_from_quantified(
    quantified: Quantified[Union[Quantifier, tuple[Quantifier, ...]]]
) -> Domain:
    qs, domain = within_domain(quantified)
    if isinstance(qs, tuple):
        idens = [q.identifier for q in qs]
    else:
        idens = [qs.identifier]
    if not _isomorphic(idens, domain.quantifiers):
        raise Exception(
            f"Inconsistent quantifiers: {idens} != {domain.quantifiers}"
        )
    return dataclasses.replace(domain, quantifiers=idens)


def _isomorphic(
    qs1: Iterable[QuantifierIdentifier], qs2: Iterable[QuantifierIdentifier]
) -> bool:
    return collections.Counter(qs1) == collections.Counter(qs2)


def cross(
    *quantifiables: Quantifiable, names: Optional[Iterable[Name]] = None
) -> Quantification:
    """Generates a cross-product quantifiable

    Args:
        quantifiables: One or more quantifiables
        names: Optional name prefixes
    """
    names_by_index = dict(enumerate(names or []))
    yield tuple(
        Quantifier(declare(q.child(name=names_by_index.get(i))))
        for i, q in enumerate(_quantifiable_quantifiers(quantifiables))
    )


def _quantifiable_quantifiers(
    quantifiable: Quantifiable,
) -> Iterable[QuantifierIdentifier]:
    if isinstance(quantifiable, tuple):
        for component in quantifiable:
            yield from _quantifiable_quantifiers(component)
    elif isinstance(quantifiable, (ScalarQuantifiable, QuantifiableReference)):
        yield QuantifierIdentifier.root(quantifiable)
    else:  # Quantification or domain
        if isinstance(quantifiable, Domain):
            domain = quantifiable
        else:
            domain = _domain_from_quantified(cast(Any, quantifiable))
        if domain.mask is not None:
            declare(domain.mask)
        yield from domain.quantifiers


def total(body: Quantified[Expression]) -> Expression:
    return _SummationExpression(*within_domain(body))


def size(quantifiable: Quantifiable) -> Expression:
    domain = domain_from_quantifiable(quantifiable)
    return _CardinalityExpression(domain)


def switch(
    *cases: Union[tuple[Predicate, ExpressionLike], ExpressionLike]
) -> Expression:
    cs: list[_SwitchCase] = []
    for t in cases:
        if isinstance(t, tuple):
            cs.append(_SwitchCase(to_expression(t[1]), t[0]))
        else:
            cs.append(_SwitchCase(to_expression(t)))
    return _SwitchExpression(cs)
