from __future__ import annotations

import collections
import dataclasses
import itertools
import math
from typing import Any, cast, Iterable, Optional, Sequence, TypeVar, Union

from ...common import untuple
from .identifiers import (
    AliasIdentifier,
    Identifier,
    Name,
    local_formatting_scope,
    QuantifierGroup,
    QuantifierIdentifier,
)
from .quantified import Quantified, unquantify, declare


def render_identifier(iden: Identifier, *subscripts: Expression) -> str:
    s = iden.format()
    if subscripts:
        sub = ",".join(s.render() for s in subscripts)
        s += f"_{{{sub}}}"
    return s


# https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class Expression:
    """Base expression

    Expressions are typically created via :class:`.Parameter` or
    :class:`.Variable` instances. They can be combined using any of the
    following operations:

    * Addition: `x + y`
    * Substraction: `x - y`
    * Multiplication: `x * y`
    * Modulo: `x % y`
    * Integer division: `x // y`
    * Floating division: `x / y`
    * Power: `x ** y`

    Literal numbers will automatically be promoted to expressions when used in
    any of the operations above. They may also be manually wrapped with
    :func:`literal`.

    Other types of expressions may be created by one of the functions below:

    * :func:`total` for summations
    * :func:`cross` for quantifiers
    * :func:`size` for cardinality expressions
    * :func:`switch` for branching logic

    See also :class:`.Predicate` for the list of supported comparison
    operators.
    """

    def __add__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("add", self, to_expression(other))

    def __radd__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) + self

    def __sub__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("sub", self, to_expression(other))

    def __rsub__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) - self

    def __mod__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("mod", self, to_expression(other))

    def __rmod__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) % self

    def __mul__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("mul", self, to_expression(other))

    def __rmul__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) * self

    def __truediv__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("div", self, to_expression(other))

    def __rtruediv__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) / self

    def __floordiv__(self, other: ExpressionLike) -> Expression:
        inner = _BinaryExpression("div", self, to_expression(other))
        return _UnaryExpression("floor", inner)

    def __rfloordiv__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) // self

    def __pow__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("pow", self, to_expression(other))

    def __rpow__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) ** self

    def __lt__(self, other: ExpressionLike) -> Predicate:
        return _ComparisonPredicate("<", self, to_expression(other))

    def __le__(self, other: ExpressionLike) -> Predicate:
        return _ComparisonPredicate("\\leq", self, to_expression(other))

    def __eq__(self, other: object) -> Predicate:  # type: ignore[override]
        if not isinstance(other, (Expression, float, int)):
            return NotImplemented
        return _ComparisonPredicate("=", self, to_expression(other))

    def __ne__(self, other: object) -> Predicate:  # type: ignore[override]
        if not isinstance(other, (Expression, float, int)):
            return NotImplemented
        return _ComparisonPredicate("\\neq", self, to_expression(other))

    def __gt__(self, other: ExpressionLike) -> Predicate:
        return _ComparisonPredicate(">", self, to_expression(other))

    def __ge__(self, other: ExpressionLike) -> Predicate:
        return _ComparisonPredicate("\\geq", self, to_expression(other))

    def __bool__(self) -> bool:
        return bool(self != 0)

    def render(self, _precedence=0) -> str:
        raise NotImplementedError()


ExpressionLike = Union[Expression, float, int]


@dataclasses.dataclass(eq=False, frozen=True)
class _LiteralExpression(Expression):
    value: float

    def render(self, _precedence=0) -> str:
        if self.value == math.inf:
            return "\\infty"
        elif self.value == -math.inf:
            return "-\\infty"
        else:
            return str(self.value)


def literal(val: Union[float, int]) -> Expression:
    """Wraps a literal value into an expression

    Arg:
        val: float or integer

    In general you will not need to use this method as expression operators
    automatically call this under the hood.
    """
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
    raise TypeError(f"Unexpected expression: {val}")


@dataclasses.dataclass(eq=False, frozen=True)
class ExpressionReference(Expression):
    identifier: Identifier
    subscripts: tuple[Expression, ...]

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
        outer = itertools.groupby(self.quantifiers, _quantifier_grouping_key)
        for key, outer_qs in outer:
            if isinstance(key, tuple):
                iden, subs = key
                inner = itertools.groupby(outer_qs, lambda q: q.outer_group)
                components: list[str] = []
                for g, inner_qs in inner:
                    group_names = list(q.format() for q in inner_qs)
                    joined = ", ".join(group_names)
                    components.append(
                        f"({joined})" if len(group_names) > 1 else joined
                    )
                group = ", ".join(components)
                group += f" \\in {render_identifier(iden, *subs)}"
                groups.append(group)
            else:
                names = ", ".join(q.format() for q in outer_qs)
                groups.append(f"{names} \\in {key.render()}")
        rendered = ", ".join(groups)
        if self.mask is not None:
            rendered += f" \\mid {self.mask.render()}"
        return rendered


def _quantifier_grouping_key(
    q: QuantifierIdentifier,
) -> Union[Space, tuple[AliasIdentifier, tuple[Expression, ...]]]:
    sp = q.space
    if not isinstance(sp, Space):
        raise TypeError(f"Unexpected space: {sp}")
    if not q.groups:
        return sp
    g = q.groups[0]
    return (g.alias, g.subscripts)


@dataclasses.dataclass(eq=False, frozen=True)
class _SummationExpression(Expression):
    summand: Expression
    domain: Domain

    def render(self, precedence=0) -> str:
        inner = max(3, precedence)
        with local_formatting_scope(self.domain.quantifiers):
            rendered = f"\\sum_{{{self.domain.render()}}} "
            rendered += self.summand.render(inner)
        return rendered


@dataclasses.dataclass(eq=False, frozen=True)
class _CardinalityExpression(Expression):
    domain: Domain

    def render(self, _precedence=0) -> str:
        with local_formatting_scope(self.domain.quantifiers):
            return f"\\lvert \\{{ {self.domain.render()} \\}} \\rvert"


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


class Space:
    def __iter__(self) -> Quantified[Quantifier]:
        return (untuple(t) for t in cross(self))

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class QuantifiableReference(Space):
    identifier: AliasIdentifier
    subscripts: tuple[Expression, ...]
    quantifiers: tuple[QuantifierIdentifier, ...]

    def render(self) -> str:
        return render_identifier(self.identifier, *self.subscripts)


@dataclasses.dataclass(frozen=True)
class Quantifier(Expression):
    """An expression used to index a quantifiable space

    Quantifiers are generated by using :func:`cross` and its convenience
    alternatives (for example iterating on a :class:`Dimension`). You should
    not need to instantiate them directly - this class is only exposed for
    typing purposes.
    """

    identifier: QuantifierIdentifier

    def render(self, _precedence=0) -> str:
        return self.identifier.format()


Quantification = Quantified[tuple[Quantifier, ...]]
"""Generic quantification result"""


def expression_space(expr: Expression) -> Optional[Space]:
    """Returns the underlying scalar quantifiable for an expression if any"""
    if isinstance(expr, Quantifier):
        return expr.identifier.space
    return None


class Predicate:
    """A predicate on expressions

    Instances of this class are generated by using comparison operators on
    :class:`.Expression` instances. The example below shows the three types of
    predicates supported as constraints:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            cost = Parameter(products)
            count = Variable(products)

            @constraint
            def total_cost_is_at_most_100(self):
                total_cost = total(
                    self.cost(p) * self.count(p)
                    for p in self.products
                )
                yield total_cost <= 100  # LEQ predicate

            @constraint
            def each_count_is_at_least_10(self):
                for p in self.products:
                    yield self.count(p) >= 10  # GEQ predicate

            @constraint
            def exactly_10_of_expensive_items(self):
                for p in self.products:
                    if self.cost(p) > 50:  # See below
                        yield self.count(p) == 10  # EQ predicate

    Additional types of predicates may be generated on expressions which do not
    include variables and used as conditionals within a quantified expression.
    For example the `exactly_10_of_expensive_items` constraint above uses a
    greater than predicate to filter the set of products where the constraint
    applies.
    """

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
        inner = _binary_condition_precedences[cond]
        left = self.left_predicate.render(inner)
        right = self.right_predicate.render(inner)
        rendered = f"{left} \\l{cond} {right}"
        if inner < precedence:
            rendered = f"\\left({rendered}\\right)"
        return rendered


Quantifiable = Union[
    Quantification,
    Quantified[Quantifier],
    Space,
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
            if mask is None:
                mask = declaration
            else:
                mask = _BinaryPredicate("and", mask, declaration)
        elif isinstance(declaration, QuantifierIdentifier):
            quantifiers.append(declaration)
        else:
            raise TypeError(f"Unexpected declaration: {declaration}")
    domain = Domain(quantifiers, mask)
    return value, domain


def domain_from_quantifiable(
    quantifiable: Quantifiable,
    names: Optional[Iterable[Name]] = None,
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
    *quantifiables: Quantifiable,
    names: Optional[Iterable[Name]] = None,
) -> Quantification:
    """Generates the cross-product of multiple quantifiables

    Args:
        quantifiables: One or more quantifiables
        names: Optional names for the generated quantifiers

    This function is the core building block for quantifying values.
    """
    names_by_index = dict(enumerate(names or []))
    yield tuple(
        Quantifier(declare(q.named(names_by_index.get(i))))
        for i, q in enumerate(_quantifiable_quantifiers(quantifiables))
    )


def _quantifiable_quantifiers(
    quantifiable: Quantifiable,
) -> Iterable[QuantifierIdentifier]:
    if isinstance(quantifiable, tuple):
        for component in quantifiable:
            yield from _quantifiable_quantifiers(component)
    elif isinstance(quantifiable, QuantifiableReference):
        qs = quantifiable.quantifiers
        group = QuantifierGroup(
            alias=quantifiable.identifier,
            subscripts=quantifiable.subscripts,
            rank=len(qs),
        )
        for q in qs:
            yield q.grouped_within(group)
    elif isinstance(quantifiable, Space):
        yield QuantifierIdentifier.base(quantifiable)
    else:  # Quantification or domain
        if isinstance(quantifiable, Domain):
            domain = quantifiable
        else:
            domain = _domain_from_quantified(cast(Any, quantifiable))
        if domain.mask is not None:
            declare(domain.mask)
        yield from domain.quantifiers


def total(body: Quantified[Expression]) -> Expression:
    """Returns an expression representing the sum of the quantified input

    Args:
        body: A quantified expression to be summed over
    """
    return _SummationExpression(*within_domain(body))


def size(quantifiable: Quantifiable) -> Expression:
    """Returns the cardinality of the quantifiable as an expression"""
    domain = domain_from_quantifiable(quantifiable)
    return _CardinalityExpression(domain)


def switch(
    *cases: Union[tuple[Predicate, ExpressionLike], ExpressionLike]
) -> Expression:
    """Returns an expression allowing branching between different values

    Args:
        cases: Tuples of expressions and optionally, predicate. The expression
            for the first predicate that matches will be used. A missing
            predicate always matches.
    """
    cs: list[_SwitchCase] = []
    for t in cases:
        if isinstance(t, tuple):
            cs.append(_SwitchCase(to_expression(t[1]), t[0]))
        else:
            cs.append(_SwitchCase(to_expression(t)))
    return _SwitchExpression(cs)
