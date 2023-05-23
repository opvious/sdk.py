from __future__ import annotations

import backoff
from datetime import datetime, timezone
import json
import humanize
import logging
from typing import cast, Iterable, Mapping, Optional, Union

from ..common import (
    Json,
    Setting,
    default_api_url,
    default_hub_url,
    format_percent,
    json_dict,
)
from ..data.attempts import (
    Attempt,
    attempt_from_graphql,
    AttemptNotification,
    notification_from_graphql,
)
from ..data.outcomes import (
    CancelledOutcome,
    failed_outcome_from_graphql,
    FeasibleOutcome,
    feasible_outcome_from_graphql,
    InfeasibleOutcome,
    Outcome,
    outcome_status,
    UnboundedOutcome,
    UnexpectedOutcomeError,
)
from ..data.outlines import Label, Outline, outline_from_json
from ..data.solves import (
    SolveInputs,
    SolveOptions,
    SolveOutputs,
    SolveResponse,
    SolveStrategy,
    solve_options_to_json,
    solve_response_from_json,
    solve_strategy_to_json,
    solve_summary_from_json,
)
from ..data.tensors import DimensionArgument, TensorArgument
from ..executors import (
    default_executor,
    Executor,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
)
from ..specifications import (
    FormulationSpecification,
    LocalSpecification,
    RemoteSpecification,
    Specification,
    local_specification_issue_from_json,
)
from ..transformations import Transformation
from .common import (
    feasible_outcome_details,
    log_progress,
    OutlineGenerator,
    SolveInputsBuilder,
)


_logger = logging.getLogger(__name__)


class Client:
    """Opvious API client"""

    def __init__(self, executor: Executor, api_url: str, hub_url: str):
        self._executor = executor
        self._api_url = api_url
        self._hub_url = hub_url

    def __repr__(self) -> str:
        fields = [
            f"executor_class={self._executor.__class__.__name__}",
            f"api_url={json.dumps(self._api_url)}",
        ]
        return f"<opvious.Client {' '.join(fields)}>"

    @classmethod
    def from_token(cls, token: str, domain: Optional[str] = None) -> Client:
        """
        Creates a client from an API token

        Args:
            token: API token. You can use an empty token to create an
                unauthenticated client with limited functionality.
            domain: API domain. You should only need to set this if you are
                using a self-hosted cluster. Defaults to the default production
                endpoint.
        """
        token = token.strip()
        authorization = None
        if token:
            authorization = token if " " in token else f"Bearer {token}"
        api_url = default_api_url(domain)
        return Client(
            executor=default_executor(
                root_url=api_url,
                authorization=authorization,
            ),
            api_url=api_url,
            hub_url=default_hub_url(domain),
        )

    @classmethod
    def from_environment(
        cls, env: Optional[dict[str, str]] = None, require_authenticated=False
    ) -> Client:
        """Creates a client from environment variables

        Args:
            env: Environment, defaults to `os.environ`.
            require_authenticated: Throw if the environment does not include a
                valid API token.
        """
        token = Setting.TOKEN.read(env)
        if not token and require_authenticated:
            raise Exception(
                f"Missing or empty {Setting.TOKEN.value} environment variable"
            )
        return Client.from_token(token=token, domain=Setting.DOMAIN.read(env))

    @property
    def authenticated(self) -> bool:
        """Returns true if the client was created with a non-empty API token"""
        return self._executor.authenticated

    @property
    def executor(self) -> Executor:
        """Returns the client's underlying executor"""
        return self._executor

    async def annotate_specification(
        self,
        specification: LocalSpecification,
        ignore_codes: Optional[Iterable[str]] = None,
    ) -> LocalSpecification:
        codes = set(ignore_codes or [])
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url="/sources/parse",
            method="POST",
            json_data=json_dict(
                sources=[s.text for s in specification.sources]
            ),
        ) as res:
            data = res.json_data()
            issues = [
                local_specification_issue_from_json(e)
                for e in data["errors"]
                if not e["code"] in codes
            ]
        return specification.annotated(issues)

    async def save_specification(
        self,
        _specification: LocalSpecification,
    ) -> None:
        raise NotImplementedError()

    async def _prepare_solve(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        transformations: Optional[list[Transformation]] = None,
        strategy: Optional[SolveStrategy] = None,
        options: Optional[SolveOptions] = None,
    ) -> tuple[Json, Outline]:
        """Generates solve candidate and final outline."""
        # First we fetch the outline to validate/coerce inputs later on
        if isinstance(specification, FormulationSpecification):
            outline_generator, tag = await OutlineGenerator.formulation(
                executor=self._executor,
                specification=specification,
            )
            formulation = json_dict(
                name=specification.formulation_name,
                specification_tag_name=tag,
            )
        else:
            if isinstance(specification, RemoteSpecification):
                sources = await specification.fetch_sources(self._executor)
            else:
                sources = [s.text for s in specification.sources]
            formulation = json_dict(sources=sources)
            outline_generator = await OutlineGenerator.sources(
                executor=self._executor, sources=sources
            )

        # Then we apply any transformations and refresh the outline if needed
        for tf in transformations or []:
            outline_generator.add_transformation(tf)
        outline, transformation_data = await outline_generator.generate()

        # Then we assemble the inputs
        builder = SolveInputsBuilder(outline=outline)
        if dimensions:
            for label, dim in dimensions.items():
                builder.set_dimension(label, dim)
        if parameters:
            for label, param in parameters.items():
                builder.set_parameter(label, param)
        inputs = builder.build()
        _logger.info(
            "Validated inputs. [parameters=%s]",
            builder.parameter_entry_count,
        )

        # Finally we put everything together
        candidate = json_dict(
            formulation=formulation,
            inputs=json_dict(
                dimensions=inputs.raw_dimensions,
                parameters=inputs.raw_parameters,
            ),
            transformations=transformation_data,
            strategy=solve_strategy_to_json(strategy, outline),
            options=solve_options_to_json(options),
        )
        return (candidate, outline)

    async def inspect_solve_instructions(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        transformations: Optional[list[Transformation]] = None,
        strategy: Optional[SolveStrategy] = None,
        options: Optional[SolveOptions] = None,
    ) -> str:
        """Returns the model's representation in `LP format`_

        The arguments below are identical to :meth:`.Client.run_solve`, making
        it easy to swap one call for another when debugging.

        Args:
            specification: :ref:`Model specification <Specifications>`
            parameters: Input data, keyed by parameter label. Values may be any
                value accepted by :meth:`.Tensor.from_argument` and must match
                the corresponding parameter's definition.
            dimensions: Dimension items, keyed by dimension label. If omitted,
                these will be automatically inferred from the parameters.
            transformations: :ref:`Model transformations`
            strategy: :ref:`Multi-objective strategy <Multi-objective
                strategies>`
            options: Solve options

        The LP formatted output will be fully annotated with matching keys and
        labels:

        .. code::

            minimize
              +1 inventory$1 \\ [day=0]
              +1 inventory$2 \\ [day=1]
              +1 inventory$3 \\ [day=2]
              \\ ...
            subject to
             inventoryPropagation$1: \\ [day=1]
              +1 inventory$1 \\ [day=1]
              -1 inventory$2 \\ [day=0]
              -1 production$1 \\ [day=1]
              = -29
             inventoryPropagation$2: \\ [day=2]
              -1 inventory$1 \\ [day=1]
              +1 inventory$3 \\ [day=2]
              -1 production$2 \\ [day=2]
              = -36
             \\ ...

        .. _LP format: https://web.mit.edu/lpsolve/doc/CPLEX-format.htm
        """
        candidate, _outline = await self._prepare_solve(
            specification=specification,
            parameters=parameters,
            dimensions=dimensions,
            transformations=transformations,
            options=options,
        )
        async with self._executor.execute(
            result_type=PlainTextExecutorResult,
            url="/solves/inspect/instructions",
            method="POST",
            json_data=json_dict(candidate=candidate),
        ) as res:
            lines = []
            async for line in res.lines():
                if line.startswith("\\"):
                    _logger.debug(line[2:].strip())
                else:
                    lines.append(line)
            return "".join(lines)

    async def run_solve(
        self,
        specification: Specification,
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        transformations: Optional[list[Transformation]] = None,
        strategy: Optional[SolveStrategy] = None,
        options: Optional[SolveOptions] = None,
        assert_feasible=False,
        prefer_streaming=True,
    ) -> SolveResponse:
        """Solves an optimization problem remotely

        Inputs will be validated before being sent to the API for solving.

        Args:
            specification: :ref:`Model specification <Specifications>`
            parameters: Input data, keyed by parameter label. Values may be any
                value accepted by :meth:`.Tensor.from_argument` and must match
                the corresponding parameter's definition.
            dimensions: Dimension items, keyed by dimension label. If omitted,
                these will be automatically inferred from the parameters.
            transformations: :ref:`Model transformations`
            strategy: :ref:`Multi-objective strategy <Multi-objective
                strategies>`
            options: Solve options
            assert_feasible: Throw if the final outcome was not feasible
            prefer_streaming: Show real time progress notifications when
                possible

        The returned response exposes both metadata (status, objective value,
        etc.) and solution data (if the solve was feasible):

        .. code:: python

            response = await client.run_solve(
                specification=opvious.RemoteSpecification.example(
                    "porfolio-selection"
                ),
                parameters={
                    "covariance": {
                        ("AAPL", "AAPL"): 0.2,
                        ("AAPL", "MSFT"): 0.1,
                        ("MSFT", "AAPL"): 0.1,
                        ("MSFT", "MSFT"): 0.25,
                    },
                    "expectedReturn": {
                        "AAPL": 0.15,
                        "MSFT": 0.2,
                    },
                    "desiredReturn": 0.1,
                },
                assert_feasible=True,  # Throw if not feasible
            )

            # Metadata is available on `outcome`
            print(f"Objective value: {response.outcome.objective_value}")

            # Solution data is available via `outputs`
            optimal_allocation = response.outputs.variable("allocation")


        See also :meth:`.Client.start_attempt` for an alternative for
        long-running solves.
        """
        candidate, outline = await self._prepare_solve(
            specification=specification,
            parameters=parameters,
            dimensions=dimensions,
            transformations=transformations,
            strategy=strategy,
            options=options,
        )

        if prefer_streaming and self._executor.supports_streaming:
            summary = None
            response_json = None
            async with self._executor.execute(
                result_type=JsonSeqExecutorResult,
                url="/solves/run",
                method="POST",
                json_data=json_dict(candidate=candidate),
            ) as res:
                async for data in res.json_seq_data():
                    kind = data["kind"]
                    if kind == "reifying":
                        progress = data["progress"]
                        if progress["kind"] == "constraint":
                            summary = progress["summary"]
                            _logger.debug(
                                "Reified constraint %r. [columns=%s, rows=%s]",
                                summary["label"],
                                summary["columnCount"],
                                summary["rowCount"],
                            )
                    elif kind == "reified":
                        summary = solve_summary_from_json(data["summary"])
                        _logger.info(
                            "Solving problem... [columns=%s, rows=%s]",
                            summary.column_count,
                            summary.row_count,
                        )
                    elif kind == "solving":
                        log_progress(_logger, data["progress"])
                    elif kind == "solved":
                        _logger.debug("Downloaded outputs.")
                        response_json = data
                    elif kind == "error":
                        message = data["error"]["message"]
                        raise Exception(f"Solve failed: {message}")
                    else:
                        raise Exception(
                            f"Unexpected response: {json.dumps(data)}"
                        )
            if not summary or not response_json:
                raise Exception("Streaming solve terminated early")
            response = solve_response_from_json(
                outline=outline,
                response_json=response_json,
                summary=summary,
            )
        else:
            async with self._executor.execute(
                result_type=JsonExecutorResult,
                url="/solves/run",
                method="POST",
                json_data=json_dict(candidate=candidate),
            ) as res:
                response = solve_response_from_json(
                    outline=outline,
                    response_json=res.json_data(),
                )

        outcome = response.outcome
        if isinstance(outcome, FeasibleOutcome):
            details = feasible_outcome_details(outcome)
            _logger.info(
                "Solve completed with status %s.%s",
                response.status,
                f" [{details}]" if details else "",
            )
        elif assert_feasible:
            raise UnexpectedOutcomeError(response.outcome)
        else:
            _logger.info("Solve completed with status %s.", response.status)

        return response

    def _attempt_url(self, uuid) -> str:
        return self._hub_url + "/attempts/" + uuid

    async def start_attempt(
        self,
        specification: Union[str, FormulationSpecification],
        parameters: Optional[Mapping[Label, TensorArgument]] = None,
        dimensions: Optional[Mapping[Label, DimensionArgument]] = None,
        transformations: Optional[list[Transformation]] = None,
        strategy: Optional[SolveStrategy] = None,
        options: Optional[SolveOptions] = None,
    ) -> Attempt:
        """Starts a new asynchronous remote solve attempt

        Inputs will be validated locally before the request is sent to the API.
        From then on, he attempt will be queued and begin solving start as soon
        as enough capacity is available.

        Args:
            specification: Model :class:`.FormulationSpecification` or
                formulation name
            parameters: Input data, keyed by parameter label. Values may be any
                value accepted by :meth:`.Tensor.from_argument` and must match
                the corresponding parameter's definition.
            dimensions: Dimension items, keyed by dimension label. If omitted,
                these will be automatically inferred from the parameters.
            transformations: :ref:`Model transformations`
            strategy: :ref:`Multi-objective strategy <Multi-objective
                strategies>`
            options: Solve options

        The returned :class:`Attempt` instance can be used to:

        + track progress via :meth:`Client.poll_attempt`,
        + retrieve inputs via :meth:`Client.fetch_attempt_inputs`,
        + retrieve outputs via :meth:`Client.fetch_attempt_outputs` (after
          successful completion).

        As a convenience, :meth:`Client.wait_for_outcome` allows polling an
        attempt until until it completes, backing off exponentially between
        each poll:

        .. code-block:: python

            # Queue a new Sudoku solve attempt
            attempt = await client.start_attempt(
                specification="sudoku",
                parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
            )

            # Wait for the attempt to complete
            await client.wait_for_outcome(
                attempt,
                assert_feasible=True  # Throw if not feasible
            )

            # Fetch the solution's data
            output_data = await client.fetch_attempt_outputs(attempt)

            # Get a parsed variable as a dataframe
            decisions = output_data.variable("decisions")

        See also :meth:`.Client.run_solve` for an alternative for short solves.
        """
        if isinstance(specification, str):
            specification = FormulationSpecification(
                formulation_name=specification
            )
        candidate, outline = await self._prepare_solve(
            specification=specification,
            parameters=parameters,
            dimensions=dimensions,
            transformations=transformations,
            strategy=strategy,
            options=options,
        )
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url="/attempts/start",
            method="POST",
            json_data=json_dict(candidate=candidate),
        ) as res:
            uuid = res.json_data()["uuid"]
        return Attempt(
            uuid=uuid,
            started_at=datetime.now(timezone.utc),
            outline=outline,
            url=self._attempt_url(uuid),
        )

    async def load_attempt(self, uuid: str) -> Optional[Attempt]:
        """Loads an existing attempt

        Args:
            uuid: The target attempt's ID
        """
        data = await self._executor.execute_graphql_query(
            query="@FetchAttempt",
            variables=json_dict(uuid=uuid),
        )
        attempt = data["attempt"]
        if not attempt:
            return None
        return attempt_from_graphql(
            data=attempt,
            outline=outline_from_json(attempt["outline"]),
            url=self._attempt_url(uuid),
        )

    async def cancel_attempt(self, uuid: str) -> bool:
        """Cancels a running attempt

        This method will throw if the attempt does not exist or is not pending
        anymore.

        Args:
            uuid: The target attempt's ID
        """
        data = await self._executor.execute_graphql_query(
            query="@CancelAttempt",
            variables=json_dict(uuid=uuid),
        )
        return bool(data["cancelAttempt"])

    async def poll_attempt(
        self, attempt: Attempt
    ) -> Union[AttemptNotification, Outcome]:
        """Polls an attempt for its outcome or latest progress notification

        Args:
            attempt: The target attempt
        """
        data = await self._executor.execute_graphql_query(
            query="@PollAttempt",
            variables=json_dict(uuid=attempt.uuid),
        )
        attempt_data = data["attempt"]
        status = attempt_data["status"]
        if status == "PENDING":
            edges = attempt_data["notifications"]["edges"]
            return notification_from_graphql(
                dequeued=bool(attempt_data["dequeuedAt"]),
                data=edges[0]["node"] if edges else None,
            )
        if status == "CANCELLED":
            return cast(Outcome, CancelledOutcome())
        if status == "INFEASIBLE":
            return InfeasibleOutcome()
        if status == "UNBOUNDED":
            return UnboundedOutcome()
        outcome = attempt_data["outcome"]
        if status == "ERRORED":
            return failed_outcome_from_graphql(outcome)
        if status == "FEASIBLE" or status == "OPTIMAL":
            return feasible_outcome_from_graphql(outcome)
        raise Exception(f"Unexpected status {status}")

    @backoff.on_predicate(
        backoff.fibo,
        lambda ret: ret is None,
        max_value=90,
        logger=None,
    )
    async def _track_attempt(self, attempt: Attempt) -> Optional[Outcome]:
        ret = await self.poll_attempt(attempt)
        if isinstance(ret, AttemptNotification):
            delta = datetime.now(timezone.utc) - attempt.started_at
            elapsed = humanize.naturaldelta(delta, minimum_unit="milliseconds")
            if ret.dequeued:
                details = [f"elapsed={elapsed}"]
                if ret.relative_gap is not None:
                    details.append(f"gap={format_percent(ret.relative_gap)}")
                if ret.cut_count is not None:
                    details.append(f"cuts={ret.cut_count}")
                if ret.lp_iteration_count is not None:
                    details.append(f"iterations={ret.lp_iteration_count}")
                _logger.info("Attempt is running... [%s]", ", ".join(details))
            else:
                _logger.info("Attempt is queued... [elapsed=%s]", elapsed)
            return None
        return ret

    async def wait_for_outcome(
        self,
        attempt: Attempt,
        assert_feasible=False,
    ) -> Outcome:
        """Waits for the attempt to complete and returns its outcome

        This method emits real-time progress messages as INFO logs.

        Args:
            attempt: The target attempt
            assert_feasible: Throw if the final outcome was not feasible
        """
        outcome = await self._track_attempt(attempt)
        if not outcome:
            raise Exception("Missing outcome")
        status = outcome_status(outcome)
        if isinstance(outcome, FeasibleOutcome):
            details = feasible_outcome_details(outcome)
            _logger.info(
                "Attempt completed with status %s.%s",
                status,
                f" [{details}]" if details else "",
            )
        elif assert_feasible:
            raise UnexpectedOutcomeError(outcome)
        else:
            _logger.info("Attempt completed with status %s.", status)
        return outcome

    async def fetch_attempt_inputs(self, attempt: Attempt) -> SolveInputs:
        """Retrieves an attempt's inputs

        Args:
            attempt: The target attempt
        """
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url=f"/attempts/{attempt.uuid}/inputs",
        ) as res:
            data = res.json_data()
        return SolveInputs(
            outline=attempt.outline,
            raw_parameters=data["parameters"],
            raw_dimensions=data["dimensions"],
        )

    async def fetch_attempt_outputs(self, attempt: Attempt) -> SolveOutputs:
        """Retrieves a successful attempt's outputs

        This method will throw if the attempt did not have a feasible solution.

        Args:
            attempt: The target attempt
        """
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url=f"/attempts/{attempt.uuid}/outputs",
        ) as res:
            data = res.json_data()
        return SolveOutputs(
            outline=attempt.outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )
