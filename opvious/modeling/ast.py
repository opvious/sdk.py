from __future__ import annotations

import dataclasses
from typing import Iterable, Optional, Tuple, TypeVar, Union

from .identifier import HasIdentifier, Identifier
from .lazy import Lazy, force, declare


# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class Expression:
    def __add__(self, other):
        return _BinaryExpression("add", self, other)

    def __radd__(self, left):
        return _BinaryExpression("add", _literal(left), self)

    def __sub__(self, other):
        return _BinaryExpression("sub", self, other)

    def __rsub__(self, left):
        return _BinaryExpression("sub", _literal(left), self)

    def __mod__(self, other):
        return _BinaryExpression("mod", self, other)

    def __rmod__(self, left):
        return _BinaryExpression("mod", _literal(left), self)

    def __mul__(self, other):
        return _BinaryExpression("mul", self, other)

    def __rmul__(self, left):
        return _BinaryExpression("mul", _literal(left), self)

    def __truediv__(self, other):
        return _BinaryExpression("div", self, other)

    def __rtruediv__(self, left):
        return _BinaryExpression("div", _literal(left), self)

    def __pow__(self, other):
        return _BinaryExpression("pow", self, other)

    def __rpow__(self, left):
        return _BinaryExpression("pow", _literal(left), self)

    def __lt__(self, other):
        return _ComparisonPredicate("<", self, other)

    def __le__(self, other):
        return _ComparisonPredicate("\\leq", self, other)

    def __eq__(self, other):
        return _ComparisonPredicate("=", self, other)

    def __ne__(self, other):
        return _ComparisonPredicate("\\neq", self, other)

    def __gt__(self, other):
        return _ComparisonPredicate(">", self, other)

    def __ge__(self, other):
        return _ComparisonPredicate("\\geq", self, other)

    def __bool__(self):
        return self != 0

    def render(self, _precedence=0):
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _LiteralExpression(Expression):
    value: float

    def render(self, _precedence=0):
        return str(self.value)


def _literal(val):
    if not isinstance(val, (float, int)):
        raise TypeError("Unexpected left operand")
    return _LiteralExpression(val)


@dataclasses.dataclass(frozen=True)
class _ReferenceExpression(Expression):
    identifier: Identifier
    subscripts: list[Expression] = dataclasses.field(
        default_factory=lambda: []
    )

    def render(self, _precedence=0):
        if not self.subscripts:
            return self.name
        subscript = ",".join(s.render() for s in self.subscripts)
        return f"{self.identifier}_{{{subscript}}}"


@dataclasses.dataclass(frozen=True)
class _UnaryExpression(Expression):
    operator: str
    expression: Expression

    def render(self, _precedence=0):
        raise NotImplementedError()  # TODO


_binary_operator_precedences = {
    "mul": (4, 4),
    "add": (1, 1),
    "mod": (3, 3),
    "sub": (2, 2),
    "div": (0, 5),
    "pow": (0, 5),
}


@dataclasses.dataclass(frozen=True)
class _BinaryExpression(Expression):
    operator: str
    left_expression: Expression
    right_expression: Expression

    def render(self, precedence=0):
        op = self.operator
        inner, outer = _binary_operator_precedences[op]
        left = self.left_expression.render(inner)
        right = self.right_expression.render(inner)
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
            rendered = f"\\left({rendered}\\right"
        return rendered


@dataclasses.dataclass(frozen=True)
class LocalIdentifier(Identifier):
    source: HasIdentifier


@dataclasses.dataclass(frozen=True)
class Domain:
    declarations: list[LocalIdentifier]
    mask: Optional[Predicate]


@dataclasses.dataclass(frozen=True)
class _SummationExpression(Expression):
    summand: Expression
    domain: Domain

    def render(self, _precedence=0):
        raise NotImplementedError()  # TODO


Source = Union[HasIdentifier, Tuple["Source", ...]]


@dataclasses.dataclass(frozen=True)
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

    def render(self, _precedence=0):
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

    def render(self, precedence=0):
        cond = self.condition
        inner = _binary_operator_precedences[cond]
        left = self.left_predicate.render(inner)
        right = self.right_predicate.render(inner)
        rendered = f"{left} \\l{cond} {right}"
        if inner < precedence:
            rendered = f"\\left({rendered}\\right)"
        return rendered


_V = TypeVar("_V", bound=Union[Expression, Predicate, None])


def locally(lazy: Lazy[_V]) -> Tuple[_V, Domain]:
    value, declarations = force(lazy)
    identifiers: list[LocalIdentifier] = []
    mask: Optional[Predicate] = None
    for declaration in declarations:
        if isinstance(declaration, Predicate):
            if mask:
                mask = _BinaryPredicate("and", mask, declaration)
            else:
                mask = declaration
        elif isinstance(declaration, LocalIdentifier):
            identifiers.append(declaration)
        else:
            raise TypeError(f"Unexpected declaration: {declaration}")
    domain = Domain(declarations, mask)
    return value, domain


def _flatten_sources(source: Source) -> Iterable[HasIdentifier]:
    if isinstance(source, tuple):
        for component in source:
            yield from _flatten_sources(component)
    else:
        yield source


Space = Lazy[Tuple[Expression, ...]]


def cross(*sources) -> Space:
    yield tuple(
        _ReferenceExpression(LocalIdentifier(s))
        for s in _flatten_sources(sources)
    )


def total(body: Lazy[Expression]) -> Expression:
    return _SummationExpression(*locally(body))


def size(space: Space) -> Expression:
    _, domain = locally(None for _t in space)
    return _CardinalityExpression(domain)