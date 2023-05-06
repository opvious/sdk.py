from __future__ import annotations

import dataclasses


# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class Expression:
    def __add__(self, other):
        return BinaryExpression("add", self, other)

    def __radd__(self, left):
        return BinaryExpression("add", _literal(left), self)

    def __sub__(self, other):
        return BinaryExpression("sub", self, other)

    def __rsub__(self, left):
        return BinaryExpression("sub", _literal(left), self)

    def __mod__(self, other):
        return BinaryExpression("mod", self, other)

    def __rmod__(self, left):
        return BinaryExpression("mod", _literal(left), self)

    def __mul__(self, other):
        return BinaryExpression("mul", self, other)

    def __rmul__(self, left):
        return BinaryExpression("mul", _literal(left), self)

    def __truediv__(self, other):
        return BinaryExpression("div", self, other)

    def __rtruediv__(self, left):
        return BinaryExpression("div", _literal(left), self)

    def __pow__(self, other):
        return BinaryExpression("pow", self, other)

    def __rpow__(self, left):
        return BinaryExpression("pow", _literal(left), self)

    def __lt__(self, other):
        return ComparisonPredicate("<", self, other)

    def __le__(self, other):
        return ComparisonPredicate("\\leq", self, other)

    def __eq__(self, other):
        return ComparisonPredicate("=", self, other)

    def __ne__(self, other):
        return ComparisonPredicate("\\neq", self, other)

    def __gt__(self, other):
        return ComparisonPredicate(">", self, other)

    def __ge__(self, other):
        return ComparisonPredicate("\\geq", self, other)

    def render(self, precedence):
        raise NotImplementedError()


def _literal(val):
    if not isinstance(val, (float, int)):
        raise TypeError("Unexpected left operand")
    return LiteralExpression(val)


@dataclasses.dataclass
class LiteralExpression(Expression):
    value: float

    def render(self, _precedence=0):
        return str(self.value)


@dataclasses.dataclass
class ReferenceExpression(Expression):
    name: str
    anchors: list[str]

    def render(self, _precedence):
        if not self.anchors:
            return self.name
        subscript = ",".join(self.anchors)
        return f"{self.name}_{{{subscript}}}"


@dataclasses.dataclass
class UnaryExpression(Expression):
    operator: str
    expression: Expression

    def render(self, _precedence):
        raise NotImplementedError()  # TODO


_precedences = {
    "mul": 4,
    "add": 1,
    "mod": 3,  # TODO: Check
    "sub": 2,
    "div": 0,
    "pow": 0,
}


@dataclasses.dataclass
class BinaryExpression(Expression):
    operator: str
    left_expression: Expression
    right_expression: Expression

    def render(self, _precedence=0):
        raise NotImplementedError()  # TODO


@dataclasses.dataclass
class SumExpression(Expression):
    summand: Expression
    domain: Domain

    def render(self, _precedence=0):
        raise NotImplementedError()  # TODO


class Predicate:
    def render(self, precedence=0):
        raise NotImplementedError()


@dataclasses.dataclass
class ComparisonPredicate(Predicate):
    command: str
    left_expression: Expression
    right_expression: Expression

    def render(self, _precedence=0):
        left = self.left_expression.render()
        right = self.right_expression.render()
        return f"{left} {self.command} {right}"
