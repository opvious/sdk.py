from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, cast

from ..common import format_percent, Json, json_dict
from ..data.outcomes import FeasibleOutcome
from ..data.outlines import Label, Outline, outline_from_json
from ..data.solves import SolveInputs
from ..data.tensors import DimensionArgument, Tensor
from ..executors import Executor, JsonExecutorResult
from ..specifications import FormulationSpecification
from ..transformations import Transformation, TransformationContext


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
                arg, len(outline.bindings), outline.is_indicator
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
        if not self._transformations:
            return outline_from_json(self._pristine_outline_data), []

        executor = self._executor
        pristine_outline_data = self._pristine_outline_data

        class Context(TransformationContext):
            async def fetch_outline(self) -> Outline:
                async with executor.execute(
                    result_type=JsonExecutorResult,
                    url="/outlines/transform",
                    method="POST",
                    json_data=json_dict(
                        outline=pristine_outline_data,
                        transformations=self.get_json(),
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
