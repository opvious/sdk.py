"""Modeling components"""

from ..common import method_decorator
from . import fragments
from .ast import (
    Cross,
    Domain,
    Expression,
    ExpressionLike,
    IterableSpace,
    Predicate,
    Projection,
    Quantifiable,
    Quantification,
    Quantifier,
    Space,
    cross,
    domain,
    literal,
    size,
    switch,
    total,
)
from .definitions import (
    ComposeFunction,
    Constraint,
    Dimension,
    Image,
    Objective,
    Parameter,
    Tensor,
    TensorLike,
    Variable,
    alias,
    compose,
    constraint,
    infer_quantifiables,
    interval,
    objective,
)
from .model import Definition, Model, ModelFragment, Statement, relabel
from .quantified import Quantified


__all__ = [
    "Model",
    "ModelFragment",
    "Statement",
    "relabel",
    # Definitions
    "Constraint",
    "Definition",
    "Dimension",
    "Image",
    "Objective",
    "Parameter",
    "Tensor",
    "TensorLike",
    "Variable",
    "alias",
    "constraint",
    "objective",
    # Expressions and predicates
    "Expression",
    "ExpressionLike",
    "Predicate",
    "interval",
    "literal",
    "size",
    "switch",
    "total",
    # Quantification
    "Cross",
    "IterableSpace",
    "Domain",
    "Projection",
    "Quantifiable",
    "Quantification",
    "Quantified",
    "Quantifier",
    "Space",
    "cross",
    "domain",
    # Fragments
    "fragments",
    # Utilities
    "ComposeFunction",
    "compose",
    "infer_quantifiables",
    "method_decorator",
]
