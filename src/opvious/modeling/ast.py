from __future__ import annotations

import collections
import dataclasses
import itertools
import math
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

from ..common import untuple
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

    def __neg__(self) -> Expression:
        return _BinaryExpression("mul", literal(-1), self)

    def __abs__(self) -> Expression:
        return _UnaryExpression("abs", self)

    def __add__(self, other: ExpressionLike) -> Expression:
        if is_literal(self, 0):
            return to_expression(other)
        if is_literal(other, 0):
            return self
        return _BinaryExpression("add", self, to_expression(other))

    def __radd__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) + self

    def __sub__(self, other: ExpressionLike) -> Expression:
        if is_literal(self, 0):
            return -to_expression(other)
        if is_literal(other, 0):
            return self
        return _BinaryExpression("sub", self, to_expression(other))

    def __rsub__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) - self

    def __mod__(self, other: ExpressionLike) -> Expression:
        return _BinaryExpression("mod", self, to_expression(other))

    def __rmod__(self, left: ExpressionLike) -> Expression:
        return to_expression(left) % self

    def __mul__(self, other: ExpressionLike) -> Expression:
        if is_literal(self, 1):
            return to_expression(other)
        if is_literal(other, 1):
            return self
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


def is_literal(expr: ExpressionLike, val: Union[float, int]) -> bool:
    if not isinstance(expr, Expression):
        return expr == val
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
        if op == "abs":
            return f"\\lvert {self.expression.render()} \\rvert"
        return f"\\left\\l{op} {self.expression.render()} \\right\\r{op}"


_binary_operator_precedences = {
    "mul": (4, 4, 4),
    "add": (1, 1, 1),
    "mod": (3, 3, 3),
    "sub": (1, 2, 1),
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
            if is_literal(self.left_expression, -1):
                rendered = f"{{-{right}}}"
            elif is_literal(self.right_expression, -1):
                rendered = f"{{-{left}}}"
            else:
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
    quantifiers: tuple[QuantifierIdentifier, ...]
    mask: Optional[Predicate] = None

    def render(self) -> str:
        groups = []
        outer = itertools.groupby(self.quantifiers, _quantifier_grouping_key)
        for (_id, key), outer_qs in outer:
            if isinstance(key, ScalarSpace):
                names = ", ".join(q.format() for q in outer_qs)
                groups.append(f"{names} \\in {key.render()}")
            else:
                inner = itertools.groupby(outer_qs, lambda q: q.outer_group)
                components: list[str] = []
                for g, inner_qs in inner:
                    group_names = list(q.format() for q in inner_qs)
                    joined = ", ".join(group_names)
                    components.append(
                        f"({joined})" if len(group_names) > 1 else joined
                    )
                group = ", ".join(components)
                group += " \\in "
                group += render_identifier(key.alias, *key.subscripts)
                groups.append(group)
        rendered = ", ".join(groups)
        if self.mask is not None:
            rendered += f" \\mid {self.mask.render()}"
        return rendered


def _quantifier_grouping_key(
    q: QuantifierIdentifier,
) -> tuple[int, Union[ScalarSpace, QuantifierGroup]]:
    # We add the ID to prevent `__eq__` from being called on equations
    sp = q.space
    if not isinstance(sp, ScalarSpace):
        raise TypeError(f"Unexpected space: {sp}")
    if not q.groups:
        return (id(sp), sp)
    g = q.groups[0]
    return (id(g), g)


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
        qs = self.domain.quantifiers
        with local_formatting_scope(qs):
            if len(qs) == 1 and self.domain.mask is None:
                _id, key = _quantifier_grouping_key(qs[0])
                if isinstance(key, ScalarSpace):
                    sp = key.render()
                else:
                    sp = render_identifier(key.alias, *key.subscripts)
            else:
                sp = f"\\{{ {self.domain.render()} \\}}"
            return f"\\# {sp}"


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
    """Base quantification


    This class provides support for generating cross-products with the `*`
    operator (see :func:`~opvious.modeling.cross`):

    .. code-block:: python

        space1 * space2 # Equivalent to cross(space1, space2)
    """

    def __mul__(self, other: Quantifiable) -> Quantification:
        return cross(self, other)

    def __rmul__(self, left: Quantifiable) -> Quantification:
        return cross(left, self)


class ScalarSpace(Space):
    def __iter__(self) -> Quantified[Quantifier]:
        return (untuple(t) for t in cross(self))

    def render(self) -> str:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class QuantifiableReference(ScalarSpace):
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


_Q = TypeVar(
    "_Q", bound=Union[Quantifier, Sequence[Quantifier]], covariant=True
)


class IterableSpace(Protocol[_Q]):
    """Base protocol for spaces which can also be directly iterated on

    It is exposed mostly as a typing convenience for typing model fragments.
    :class:`~opvious.modeling.Space` is typically used for providing the
    underlying implementation.
    """

    def __mul__(self, other: Quantifiable) -> Quantification:
        raise NotImplementedError()

    def __rmul__(self, other: Quantifiable) -> Quantification:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[_Q]:
        raise NotImplementedError()


def expression_space(expr: Expression) -> Optional[ScalarSpace]:
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
    Iterable[Union[Quantifier, Sequence[Quantifier]]],  # Includes quantified
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
    domain = Domain(tuple(quantifiers), mask)
    return value, domain


def domain(
    quantifiable: Quantifiable,
    names: Optional[Iterable[Name]] = None,
) -> Domain:
    """Creates a domain from a quantifiable"""
    return _domain_from_quantified(iter(cross(quantifiable, names=names)))


def _domain_from_quantified(
    quantified: Quantified[Union[Quantifier, Iterable[Quantifier]]]
) -> Domain:
    qs, domain = within_domain(quantified)
    if isinstance(qs, Quantifier):
        idens: tuple[QuantifierIdentifier, ...] = (qs.identifier,)
    else:
        idens = tuple(q.identifier for q in qs)
    if not _isomorphic(idens, domain.quantifiers):
        raise Exception(
            f"Inconsistent quantifiers: {idens} != {domain.quantifiers}"
        )
    return dataclasses.replace(domain, quantifiers=idens)


def _isomorphic(
    qs1: Iterable[QuantifierIdentifier], qs2: Iterable[QuantifierIdentifier]
) -> bool:
    return collections.Counter(qs1) == collections.Counter(qs2)


Projection = int


@dataclasses.dataclass(frozen=True)
class Cross(Sequence[Quantifier]):
    """Cross-product result"""

    _quantifiers: tuple[Quantifier, ...]
    _lifted: Optional[tuple[Quantifier, ...]]

    @property
    def lifted(self) -> tuple[Quantifier, ...]:
        if self._lifted is None:
            raise Exception("Unlifted cross-product")
        return self._lifted

    def __len__(self) -> int:
        return len(self._quantifiers)

    @overload
    def __getitem__(self, ix: int) -> Quantifier:
        ...

    @overload
    def __getitem__(self, sl: slice) -> Sequence[Quantifier]:
        ...

    def __getitem__(self, arg: Any) -> Any:
        return self._quantifiers[arg]

    def __iter__(self):
        return iter(self._quantifiers)


@dataclasses.dataclass(
    frozen=True,
    eq=False,
)
class Quantification(Space):
    """Cross-product quantification"""

    _quantifiables: tuple[Quantifiable, ...]
    _names: Mapping[int, Name]
    _projection: Projection
    _lift: bool

    __hash__: Any = None

    def __iter__(self) -> Quantified[Cross]:
        projected: list[Quantifier] = []
        lifted: list[Quantifier] = []
        for i, d in enumerate(self._quantifiables):
            project = (1 << i) & self._projection
            if not project and not self._lift:
                continue
            j0 = len(projected)
            quants = list(
                Quantifier(declare(iden.named(self._names.get(j0 + j))))
                for j, iden in enumerate(_quantifier_identifiers(d))
            )
            lifted.extend(quants)
            if project:
                projected.extend(quants)
        yield Cross(tuple(projected), tuple(lifted))


def lift(
    projected: tuple[Quantifier, ...],
    unprojected: tuple[Quantifier, ...],
    projection: Projection,
) -> tuple[Quantifier, ...]:
    """Combines quantifiers to reconstruct an underlying quantification"""
    quants: list[Quantifier] = []
    i = j = 0
    for k in range(len(projected) + len(unprojected)):
        if (1 << k) & projection:
            quants.append(projected[i])
            i += 1
        else:
            quants.append(unprojected[j])
            j += 1
    return tuple(quants)


def cross(
    *quantifiables: Quantifiable,
    names: Optional[Iterable[Name]] = None,
    projection: Projection = -1,
    lift=False,
) -> Quantification:
    """Generates the cross-product of multiple quantifiables

    Args:
        quantifiables: One or more quantifiables
        names: Optional names for the generated quantifiers
        projection: Quantifiable selection mask
        lift: Returns lifted :class:`~opvious.modeling.Cross` instances.
            Setting this option will include all masks present in the original
            quantifiable, even if they are not projected.

    This function is the core building block for quantifying values.
    """
    return Quantification(
        _quantifiables=quantifiables,
        _names=dict(enumerate(names or [])),
        _projection=projection,
        _lift=lift,
    )


def _quantifier_identifiers(
    quantifiable: Quantifiable,
) -> Iterable[QuantifierIdentifier]:
    if isinstance(quantifiable, tuple):
        for component in quantifiable:
            yield from _quantifier_identifiers(component)
    elif isinstance(quantifiable, QuantifiableReference):
        qs = quantifiable.quantifiers
        group = QuantifierGroup(
            alias=quantifiable.identifier,
            subscripts=quantifiable.subscripts,
            rank=len(qs),
        )
        for q in qs:
            yield q.grouped_within(group)
    elif isinstance(quantifiable, ScalarSpace):
        yield QuantifierIdentifier.base(quantifiable)
    else:  # domain or quantified
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
    return _CardinalityExpression(domain(quantifiable))


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
