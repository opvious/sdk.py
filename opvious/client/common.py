from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, cast

from ..common import format_percent, Json, json_dict
from ..data.outcomes import FeasibleOutcome
from ..data.outlines import Label, Outline, outline_from_json
from ..data.solves import SolveInputs, SolveOptions, SolveStrategy
from ..data.tensors import DimensionArgument, Tensor, TensorArgument
from ..executors import Executor, JsonExecutorResult
from ..specifications import FormulationSpecification, Specification
from ..transformations import Transformation, TransformationContext


DEFAULT_ENDPOINT = "https://api.cloud.opvious.io"


class ClientSetting(enum.Enum):
    """Client configuration environment variables"""

    TOKEN = ("OPVIOUS_TOKEN", "")
    ENDPOINT = ("OPVIOUS_ENDPOINT", DEFAULT_ENDPOINT)

    def read(self, env: Optional[dict[str, str]] = None) -> str:
        """Read the setting's current value or default if missing

        Args:
            env: Environment, defaults to `os.environ`.
        """
        if env is None:
            env = cast(Any, os.environ)
        name, default_value = self.value
        return env.get(name) or default_value


def log_progress(logger: logging.Logger, progress: Json) -> None:
    kind = progress["kind"]
    if kind == "activity":
        iter_count = progress.get("lpIterationCount")
        gap = progress.get("relativeGap")
        if iter_count is not None:
            logger.info(
                "Solve in progress... [iterations=%s, gap=%s]",
                iter_count,
                "n/a" if gap is None else format_percent(gap),
            )
    elif kind == "epsilonConstraint":
        logger.info(
            "Added epsilon constraint. [objective_value=%s]",
            progress["objectiveValue"],
        )
    else:
        raise Exception(f"Unsupported progress kind: {kind}")


class SolveInputsBuilder:
    def __init__(self, outline: Outline):
        self._outline = outline
        self._dimensions: Dict[Label, Any] = {}
        self._parameters: Dict[Label, Any] = {}
        self.parameter_entry_count = 0

    def set_dimension(self, label: Label, arg: DimensionArgument) -> None:
        outline = self._outline.dimensions.get(label)
        if not outline:
            raise Exception(f"Unknown dimension: {label}")
        if label in self._dimensions:
            raise Exception(f"Duplicate dimension: {label}")
        items = list(arg)
        self._dimensions[label] = {"label": label, "items": items}

    def set_parameter(self, label: Label, arg: Any) -> None:
        outline = self._outline.parameters.get(label)
        if not outline:
            raise Exception(f"Unknown parameter: {label}")
        if label in self._parameters:
            raise Exception(f"Duplicate parameter: {label}")
        try:
            tensor = Tensor.from_argument(
                arg,
                rank=len(outline.bindings),
                is_indicator=outline.is_indicator,
                is_pin=outline.derivation_kind == "pinnedVariable",
            )
        except Exception as exc:
            raise ValueError(f"Invalid  parameter: {label}") from exc
        self._parameters[label] = json_dict(
            label=label,
            entries=tensor.entries,
            default_value=tensor.default_value,
        )
        self.parameter_entry_count += len(tensor.entries)

    def build(self) -> SolveInputs:
        missing_labels = set()

        for label in self._outline.parameters:
            if label not in self._parameters:
                missing_labels.add(label)

        if self._dimensions:
            for label in self._outline.dimensions:
                if label not in self._dimensions:
                    missing_labels.add(label)

        if missing_labels:
            raise Exception(f"Missing label(s): {missing_labels}")

        return SolveInputs(
            outline=self._outline,
            raw_parameters=list(self._parameters.values()),
            raw_dimensions=list(self._dimensions.values()) or None,
        )


class OutlineGenerator:
    def __init__(self, executor: Executor, outline_data: Json):
        self._executor = executor
        self._pristine_outline_data = outline_data
        self._transformations = cast(list[Transformation], [])

    @classmethod
    async def formulation(
        cls, executor: Executor, specification: FormulationSpecification
    ) -> tuple[OutlineGenerator, str]:
        data = await executor.execute_graphql_query(
            query="@FetchOutline",
            variables={
                "formulationName": specification.formulation_name,
                "tagName": specification.tag_name,
            },
        )
        formulation = data.get("formulation")
        if not formulation:
            raise Exception("No matching formulation found")
        tag = formulation.get("tag")
        if not tag:
            raise Exception("No matching specification found")
        spec = tag["specification"]
        return (OutlineGenerator(executor, spec["outline"]), tag["name"])

    @classmethod
    async def sources(
        cls, executor: Executor, sources: list[str]
    ) -> OutlineGenerator:
        async with executor.execute(
            result_type=JsonExecutorResult,
            url="/sources/parse",
            method="POST",
            json_data=json_dict(sources=sources, outline=True),
        ) as res:
            outline_data = res.json_data()
        errors = outline_data.get("errors")
        if errors:
            raise Exception(f"Invalid sources: {json.dumps(errors)}")
        return OutlineGenerator(executor, outline_data["outline"])

    def add_transformation(self, tf: Transformation) -> None:
        self._transformations.append(tf)

    async def generate(self) -> tuple[Outline, Json]:
        pristine_outline = outline_from_json(self._pristine_outline_data)
        if not self._transformations:
            return pristine_outline, []
        executor = self._executor
        pristine_outline_data = self._pristine_outline_data

        class Context(TransformationContext):
            async def fetch_outline(self) -> Outline:
                transformations = self.get_json()
                if not transformations:
                    return pristine_outline
                async with executor.execute(
                    result_type=JsonExecutorResult,
                    url="/outlines/transform",
                    method="POST",
                    json_data=json_dict(
                        outline=pristine_outline_data,
                        transformations=transformations,
                    ),
                ) as res:
                    data = res.json_data()
                return outline_from_json(data["outline"])

        context = Context()
        for tf in self._transformations:
            await tf.register(context)
        outline = await context.fetch_outline()
        return outline, context.get_json()


def feasible_outcome_details(outcome: FeasibleOutcome) -> Optional[str]:
    details = []
    if outcome.objective_value:
        details.append(f"objective={outcome.objective_value}")
    if outcome.relative_gap:
        details.append(f"gap={format_percent(outcome.relative_gap)}")
    return ", ".join(details) if details else None


@dataclasses.dataclass(frozen=True)
class Problem:
    """An optimization problem instance"""

    specification: Specification
    """:ref:`Model specification <Specifications>`"""

    parameters: Optional[Mapping[Label, TensorArgument]] = None
    """Input data, keyed by parameter label

    Values may be any value accepted by :meth:`.Tensor.from_argument` and must
    match the corresponding parameter's definition.
    """

    dimensions: Optional[Mapping[Label, DimensionArgument]] = None
    """Dimension items, keyed by dimension label

    If omitted, these will be automatically inferred from the parameters.
    """

    transformations: Optional[list[Transformation]] = None
    """:ref:`Transformations` to apply to the specification"""

    strategy: Optional[SolveStrategy] = None
    """:ref:`Multi-objective strategy <Multi-objective strategies>`"""

    options: Optional[SolveOptions] = None
    """Solve options (gap thresholds, timeout, etc.)"""
