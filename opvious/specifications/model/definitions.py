from __future__ import annotations

import dataclasses
import logging
import math
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from ...common import Label, to_camel_case
from .ast import (
    Expression,
    ExpressionLike,
    ExpressionReference,
    literal,
    Predicate,
    ScalarQuantifiable,
    Quantifiable,
    QuantifiableReference,
    Quantification,
    Quantifier,
    cross,
    domain_from_quantifiable,
    expression_quantifiable,
    is_literal,
    render_identifier,
    to_expression,
    within_domain,
)
from .identifiers import (
    AliasIdentifier,
    DimensionIdentifier,
    Environment,
    GlobalIdentifier,
    IdentifierFormatter,
    Name,
    QuantifierIdentifier,
    TensorIdentifier,
    TensorVariant,
    local_formatting_scope,
    global_formatting_scope,
)
from .images import Image
from .quantified import Quantified


_logger = logging.getLogger(__name__)


class _Definition:
    @property
    def label(self) -> Optional[Label]:
        raise NotImplementedError()

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        raise NotImplementedError()

    def render_statement(self, label: Label, model: Any) -> Optional[str]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class _Statement:
    label: Label
    definition: _Definition
    fragment: ModelFragment


class ModelFragment:
    """Model partial"""


@dataclasses.dataclass(frozen=True)
class _Relabeled(ModelFragment):
    fragment: ModelFragment
    labels: Mapping[str, Label]


def relabel(fragment: ModelFragment, **kwargs: Label) -> ModelFragment:
    """Updates a fragment's definitions' labels"""
    return _Relabeled(fragment, kwargs)


class _StatementVisitor:
    def __init__(self) -> None:
        self.statements: list[_Statement] = []
        self._visited: set[int] = set()  # Model IDs

    def visit(
        self,
        model: Model,
        omit_dependencies: bool,
        prefix: Optional[str] = None,
    ) -> None:
        model_id = id(model)
        if model_id in self._visited:
            return
        self._visited.add(model_id)

        self._visit_fragment(model, [prefix] if prefix else [])
        if not omit_dependencies:
            for dep in model.dependencies:
                self.visit(dep, False)

    def _visit_fragment(
        self, frag: ModelFragment, prefix: Sequence[str]
    ) -> None:
        labels: dict[str, Label] = {}
        while isinstance(frag, _Relabeled):
            labels.update(frag.labels)
            frag = frag.fragment

        attrs: dict[str, Any] = {}
        for cls in reversed(frag.__class__.__mro__[1:]):
            attrs.update(cls.__dict__)
        attrs.update(frag.__dict__)
        attrs.update(frag.__class__.__dict__)

        path = [*prefix, ""]
        for attr, value in attrs.items():
            path[-1] = attr
            if isinstance(value, property):
                value = value.fget
            if not isinstance(value, _Definition):
                if isinstance(value, ModelFragment):
                    self._visit_fragment(value, path)
                continue
            label = (
                labels.get(attr)
                or value.label
                or to_camel_case("_".join(path))
            )
            self.statements.append(_Statement(label, value, frag))


class ModelValidator:
    def validate(self, source: str) -> str:
        raise NotImplementedError()


class Model(ModelFragment):
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
            covers = Parameter(sets, vertices, image=indicator())
            used = Variable(sets, image=indicator())

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
    __prefix: Optional[str] = None
    __validator: Optional[ModelValidator] = None

    def __init__(
        self,
        dependencies: Optional[Iterable[Model]] = None,
        prefix: Optional[Label] = None,
    ):
        self.__dependencies = list(dependencies) if dependencies else None
        self.__prefix = prefix

    @classmethod
    def set_default_validator(cls, validator: Optional[ModelValidator]):
        """Sets the validator used when no explicit one is provided"""
        cls.__validator = validator

    @property
    def dependencies(self) -> Sequence[Model]:
        """The model's dependencies"""
        return self.__dependencies or []

    def render_specification(
        self,
        *,
        labels: Optional[Iterable[Label]] = None,
        omit_dependencies=True,
        formatter_factory: Optional[
            Callable[[Mapping[GlobalIdentifier, Label]], IdentifierFormatter]
        ] = None,
        validator: Union[ModelValidator, None] = None,
    ) -> str:
        """Generates the model's specification

        Args:
            labels: Allowlist of labels to render. If specified, only these
                definitions and transitively referenced ones will be rendered.
                By default all definitions are rendered.
            omit_dependencies: Omit any definitions which belong to this
                model's dependencies. This is useful to produce a partial
                specification which can be inspected more easily.
            formatter_factory: Custom name formatter
        """
        visitor = _StatementVisitor()
        visitor.visit(
            self,
            prefix=self.__prefix,
            omit_dependencies=omit_dependencies,
        )
        statements = visitor.statements

        allowlist = set(labels or [])
        by_identifier = {
            s.definition.identifier: s
            for s in statements
            if s.definition.identifier
        }
        labels_by_identifier = {
            i: d.label for i, d in by_identifier.items() if d.label
        }
        if formatter_factory:
            formatter = formatter_factory(labels_by_identifier)
        else:
            formatter = _DefaultFormatter(labels_by_identifier)
        reserved = {i.name: i for i in labels_by_identifier if i.name}
        with global_formatting_scope(formatter, reserved):
            idens = set()
            rendered: list[Optional[str]] = []
            for s in statements:
                if not s.label or (allowlist and s.label not in allowlist):
                    continue
                rs = s.definition.render_statement(s.label, s.fragment)
                if not rs:
                    continue
                rendered.append(rs)
                if s.definition.identifier:
                    idens.add(s.definition.identifier)
            for iden in formatter.formatted_globals():
                if iden in idens:
                    continue
                ds = by_identifier.get(iden)
                if not ds:
                    if omit_dependencies:
                        raise Exception(f"Missing statement: {iden}")
                    continue
                rs = ds.definition.render_statement(ds.label, self)
                if not rs:
                    raise Exception(f"Missing rendered statement: {iden}")
                rendered.append(rs)
            contents = "".join(f"  {s} \\\\\n" for s in rendered if s)

        source = f"$$\n\\begin{{align}}\n{contents}\\end{{align}}\n$$"
        if not validator:
            validator = self.__validator
        if validator:
            source = validator.validate(source)
        return source

    def _repr_latex_(self) -> str:
        # Magic method used by Jupyter
        return self.render_specification(omit_dependencies=True)


class _DefaultFormatter(IdentifierFormatter):
    def __init__(self, labels: Mapping[GlobalIdentifier, Label]) -> None:
        super().__init__(labels)

    def _format_dimension(self, label: Label, env: Environment) -> Name:
        return f"D^{{{label}}}"

    def _format_parameter(self, label: Label, env: Environment) -> Name:
        return f"p^{{{label}}}"

    def _format_variable(self, label: Label, env: Environment) -> Name:
        return f"v^{{{label}}}"

    def format_quantifier(
        self, identifier: QuantifierIdentifier, env: Environment
    ) -> Name:
        name = identifier.name
        if not name:
            quantifiable = identifier.quantifiable
            if isinstance(quantifiable, QuantifiableReference):
                name = quantifiable.identifier.format().lower()
        return self._first_available(name or "i", env)

    def _first_available(self, name: Name, env: Environment) -> Name:
        while name in env:
            name += "x"
        return name


class Dimension(_Definition, ScalarQuantifiable):
    """An abstract collection of values

    Args:
        label: Dimension label override. By default the label is derived from
            the attribute's name
        name: The dimension's name. By default the name will be derived from
            the dimension's label
        is_numeric: Whether the dimension will only contain integers. This
            enables arithmetic operations on this dimension's quantifiers

    Dimensions are `Quantifiable` and as such can be quantified over using
    :func:`.cross`. As a convenience, iterating on a dimension also a suitable
    quantifier. This allows creating simple constraints directly:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable(products)

            @constraint
            def at_least_one_of_each(self):
                for p in self.products:  # Note the iteration here
                    yield self.count(p) >= 1
    """

    def __init__(
        self,
        *,
        label: Optional[Label] = None,
        name: Optional[Name] = None,
        is_numeric: bool = False,
    ):
        self._identifier = DimensionIdentifier(name=name)
        self._label = label
        self._is_numeric = is_numeric

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def render(self) -> str:
        return self._identifier.format()

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        s = f"\\S^d_{{{label}}}&: {self._identifier.format()}"
        if self._is_numeric:
            s += " \\subset \\mathbb{Z}"
        return s


@dataclasses.dataclass(frozen=True)
class _Interval(ScalarQuantifiable):
    lower_bound: Expression
    upper_bound: Expression

    def render(self) -> str:
        lb = self.lower_bound
        ub = self.upper_bound
        if is_literal(lb, 0) and is_literal(ub, 1):
            return "\\{0, 1\\}"
        return f"\\{{ {lb.render()} \\ldots {ub.render()} \\}}"


_integers = _Interval(literal(-math.inf), literal(math.inf))


def interval(
    lower_bound: ExpressionLike, upper_bound: ExpressionLike
) -> Quantified[Quantifier]:
    """A range of values

    Args:
        lower_bound: The range's inclusive lower bound
        upper_bound: The range's inclusive upper bound
    """
    interval = _Interval(
        lower_bound=to_expression(lower_bound),
        upper_bound=to_expression(upper_bound),
    )
    return iter(interval)


class _Tensor(_Definition):
    def __init__(
        self,
        *quantifiables: Quantifiable,
        name: Optional[Name] = None,
        label: Optional[Label] = None,
        image: Image = Image(),
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        self._identifier = TensorIdentifier(name=name, variant=self._variant)
        self._domain = domain_from_quantifiable(quantifiables)
        self._label = label
        self.image = image
        self.qualifiers = qualifiers

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def quantification(self) -> Quantification:
        return cross(self._domain)

    @property
    def _variant(self) -> TensorVariant:
        raise NotImplementedError()

    def __call__(self, *subscripts: ExpressionLike) -> Expression:
        return ExpressionReference(
            self._identifier, tuple(to_expression(s) for s in subscripts)
        )

    def render_statement(self, label: Label, _model: Any) -> Optional[str]:
        c = self._variant[0]
        s = f"\\S^{c}_{{{_render_label(label, self.qualifiers)}}}&: "
        s += f"{self._identifier.format()} \\in "
        s += self.image.render()
        domain = self._domain
        if domain.quantifiers:
            with local_formatting_scope(domain.quantifiers):
                if domain.mask:
                    sup = domain.render()
                else:
                    formatted = [
                        q.quantifiable.render() for q in domain.quantifiers
                    ]
                    sup = " \\times ".join(formatted)
                s += f"^{{{sup}}}"
        return s


def _render_label(label: Label, qualifiers: Optional[Sequence[Label]]) -> str:
    s = label
    if qualifiers:
        s += f"[{','.join(qualifiers)}]"
    return s


class Parameter(_Tensor):
    """An optimization input parameter"""

    _variant = "parameter"


class Variable(_Tensor):
    """An optimization output variable"""

    _variant = "variable"


class _FragmentMethod:
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    def __get__(self, frag: Any, _objtype=None) -> Callable[..., Any]:
        # This is needed for non-property calls
        if not isinstance(frag, ModelFragment):
            raise TypeError(f"Unexpected owner: {frag}")

        def wrapped(*args, **kwargs) -> Any:
            return self(frag, *args, **kwargs)

        return wrapped


_M = TypeVar("_M", bound=Model, contravariant=True)


_R = TypeVar(
    "_R",
    bound=Union[Expression, Quantification, Quantified[Quantifier]],
    covariant=True,
)


class _Aliasable(Protocol[_M, _R]):
    def __call__(self, model: _M, *exprs: ExpressionLike) -> _R:
        pass


_AliasedVariant = Literal[
    "expression",
    "scalar_quantification",
    "quantification",
]


@dataclasses.dataclass(frozen=True)
class _Aliased:
    variant: _AliasedVariant
    quantifiables: Sequence[Optional[ScalarQuantifiable]]


class _Alias(_Definition, _FragmentMethod):
    label = None

    def __init__(
        self,
        aliasable: _Aliasable[Any, Any],
        name: Name,
        quantifier_names: Optional[Iterable[Name]] = None,
    ):
        super().__init__()
        self._identifier = AliasIdentifier(name=name)
        self._quantifier_names = quantifier_names
        self._aliasable = aliasable
        self._aliased: Optional[_Aliased] = None

    @property
    def identifier(self) -> Optional[GlobalIdentifier]:
        return self._identifier

    def render_statement(self, _label: Label, model: Any) -> Optional[str]:
        if self._aliased is None:
            return None  # Not used
        quantifiable = tuple(
            q or _integers for q in self._aliased.quantifiables
        )
        outer_domain = domain_from_quantifiable(
            quantifiable, quantifier_names=self._quantifier_names
        )
        expressions = [Quantifier(q) for q in outer_domain.quantifiers]
        value = self._aliasable(model, *expressions)
        s = "\\S^a&: "
        with local_formatting_scope(outer_domain.quantifiers):
            if outer_domain.quantifiers:
                s += f"\\forall {outer_domain.render()}, "
            s += render_identifier(self._identifier, *expressions)
            s += " \\doteq "
            if self._aliased.variant == "expression":
                s += value.render()
            else:
                inner_domain = domain_from_quantifiable(value)
                with local_formatting_scope(inner_domain.quantifiers):
                    if len(inner_domain.quantifiers) > 1 or inner_domain.mask:
                        s += f"\\{{ {inner_domain.render()} \\}}"
                    else:
                        s += inner_domain.quantifiers[0].quantifiable.render()
        return s

    def __call__(self, model: Any, *subscripts: ExpressionLike) -> Any:
        # This is used for property calls
        exprs = tuple(to_expression(s) for s in subscripts)
        if self._aliased is None:
            value = self._aliasable(model, *exprs)
            if isinstance(value, Expression):
                variant: _AliasedVariant = "expression"
            elif isinstance(value, tuple):
                variant = "quantification"
            else:
                variant = "scalar_quantification"
            self._aliased = _Aliased(
                variant,
                [expression_quantifiable(x) for x in exprs],
            )
        if self._aliased.variant == "expression":
            return ExpressionReference(self._identifier, exprs)
        else:
            ref = QuantifiableReference(self._identifier, exprs)
            if self._aliased.variant == "quantification":
                return cross(ref)
            else:
                return iter(ref)


_F = TypeVar("_F", bound=Callable[..., Union[Expression, Quantifiable]])


def alias(
    name: Optional[Name], quantifier_names: Optional[Iterable[Name]] = None
) -> Callable[[_F], _F]:  # TODO: Tighten argument type
    """Decorator promoting a :class:`.Model` method to a named alias

    Args:
        name: The alias' name. If `None`, no alias will be added.
        quantifier_names: Optional names to use for the alias' quantifiers

    The method can return a (potentially quantified) expression or a
    quantification and may accept any number of expression arguments. This is
    useful to make the generated specification more readable by extracting
    commonly used sub-expressions or sub-spaces.

    Finally, the decorated function may be wrapped as a property if doesn't
    have any non-`self` arguments.

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            count = Variable(products)

            @property
            @alias("t")
            def total_product_count(self):
                return total(self.count(p) for p in self.products)
    """

    def wrap(fn):
        if name is None:
            return fn
        return _Alias(fn, name=name, quantifier_names=quantifier_names)

    return wrap


ConstraintBody = Callable[[_M], Quantified[Predicate]]


class Constraint(_Definition, _FragmentMethod):
    """Optimization constraint

    Constraints are best created directly from :class:`.Model` methods via the
    :func:`.constraint` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable(products)

            @constraint
            def at_least_one(self):
                yield total(self.count(p) for p in self.products) >= 1
    """

    identifier = None

    def __init__(
        self,
        body: ConstraintBody,
        label: Optional[Label] = None,
        qualifiers: Optional[Sequence[Label]] = None,
    ):
        super().__init__()
        self._body = body
        self._label = label
        self.qualifiers = qualifiers

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

    @property
    def label(self):
        return self._label

    def render_statement(self, label: Label, frag: Any) -> Optional[str]:
        s = f"\\S^c_{{{_render_label(label, self.qualifiers)}}}&: "
        predicate, domain = within_domain(self._body(frag))
        with local_formatting_scope(domain.quantifiers):
            if domain.quantifiers:
                s += f"\\forall {domain.render()}, "
            s += predicate.render()
        return s


@overload
def constraint(body: ConstraintBody) -> Constraint:
    ...


@overload
def constraint(
    *,
    label: Optional[Label] = None,
    qualifiers: Optional[Sequence[Label]] = None,
    disabled=False,
) -> Callable[[ConstraintBody], Constraint]:
    ...


def constraint(
    body: Optional[ConstraintBody] = None,
    *,
    label: Optional[Label] = None,
    qualifiers: Optional[Sequence[Label]] = None,
    disabled=False,
) -> Any:
    """Decorator promoting a :class:`.Model` method to a :class:`.Constraint`

    Args:
        label: Constraint label override. By default the label is derived from
            the method's name.
        qualifiers: Optional list of labels used to qualify the constraint's
            quantifiers. This is useful to override the name of the colums in
            solution dataframes.

    The decorated method should accept only a `self` argument and return a
    quantified :class:`.Predicate`.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class MyModel(Model):
            # ...

            @constraint
            def ensure_something(self):
                # ...

            @constraint(label="ensuresEverything")
            def ensure_something_else(self):
                # ...
    """
    if body:
        return Constraint(body)

    def wrap(fn):
        if disabled:
            return fn
        return Constraint(fn, label=label, qualifiers=qualifiers)

    return wrap


ObjectiveSense = Literal["max", "min"]
"""Optimization direction"""


ObjectiveBody = Callable[[_M], Expression]
"""Optimization target expression"""


class Objective(_Definition, _FragmentMethod):
    """Optimization objective

    Objectives are best created directly from :class:`.Model` methods via the
    :func:`.objective` decorator.

    .. code-block:: python

        class ProductModel:
            products = Dimension()
            count = Variable(products)

            @objective
            def minimize_total_count(self):
                return total(self.count(p) for p in self.products)

    """

    identifier = None

    def __init__(
        self,
        body: ObjectiveBody,
        sense: ObjectiveSense,
        label: Optional[Label] = None,
    ):
        super().__init__()
        self._body = body
        self._sense = sense
        self._label = label

    @property
    def label(self) -> Optional[Label]:
        return self._label

    def __call__(self, *args, **kwargs):
        return self._body(*args, **kwargs)

    def render_statement(self, label: Label, frag: Any) -> Optional[str]:
        sense = self._sense
        if sense is None:
            if label.startswith("min"):
                sense = "min"
            elif label.startswith("max"):
                sense = "max"
            else:
                raise Exception(f"Missing sense for objective {label}")
        expression = to_expression(self._body(frag))
        return f"\\S^o_{{{label}}}&: \\{sense} {expression.render()}"


@overload
def objective(body: ObjectiveBody) -> Constraint:
    ...


@overload
def objective(
    *, sense: Optional[ObjectiveSense] = None, label: Optional[Label] = None
) -> Callable[[ObjectiveBody], Constraint]:
    ...


def objective(
    body: Optional[ObjectiveBody] = None,
    *,
    sense: Optional[ObjectiveSense] = None,
    label: Optional[Label] = None,
    disabled=False,
) -> Any:
    """Decorator promoting a method to an :class:`.Objective`

    Args:
        sense: Optimization direction. This may be omitted if the method name
            starts with `min` or `max`, in which case the appropriate sense
            will be inferred.
        label: Objective label override. By default the label is derived from
            the method's name.

    The decorated method should accept only a `self` argument and return an
    :class:`.Expression`, which will become the objective's optimization
    target.

    As a convenience, this decorator can be used with and without arguments:

    .. code-block:: python

        class ProductModel(Model):
            products = Dimension()
            cost = Parameter(products)
            count = Variable(products)

            @objective
            def minimize_cost(self):
                return total(
                    self.count(p) * self.cost(p)
                    for p in self.products
                )

            @objective(sense="max")
            def optimize_count(self):
                return total(self.count(p) for p in self.products)
    """

    def wrap(fn):
        if disabled:
            return fn
        method_sense = sense
        if method_sense is None:
            name = fn.__name__
            if name.startswith("min"):
                method_sense = "min"
            elif name.startswith("max"):
                method_sense = "max"
            else:
                raise Exception(f"Missing sense for objective {name}")
        return Objective(fn, sense=method_sense, label=label)

    return wrap
