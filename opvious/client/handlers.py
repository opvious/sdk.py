from __future__ import annotations

import backoff
import json
import logging
from typing import (
    AsyncIterator,
    Iterable,
    Optional,
    Sequence,
    Union,
)

from ..common import (
    Annotation,
    Json,
    encode_annotations,
    Uuid,
    format_percent,
    gather,
    json_dict,
)
from ..data.queued_solves import (
    QueuedSolve,
    queued_solve_from_graphql,
    SolveNotification,
    solve_notification_from_graphql,
)
from ..data.outcomes import (
    FailedOutcome,
    FeasibleOutcome,
    SolveOutcome,
    UnexpectedSolveOutcomeError,
    failed_outcome_from_graphql,
    solve_outcome_from_graphql,
    solve_outcome_status,
)
from ..data.outlines import ProblemOutline
from ..data.solves import (
    ProblemSummary,
    SolveInputs,
    SolveOutputs,
    Solution,
    problem_summary_from_json,
    solve_options_to_json,
    solution_from_json,
    solve_strategy_to_json,
)
from ..executors import (
    Executor,
    JsonExecutorResult,
    JsonSeqExecutorResult,
    PlainTextExecutorResult,
    authorization_header,
    default_executor,
)
from ..specifications import (
    FormulationSpecification,
    LocalSpecification,
    RemoteSpecification,
    local_specification_issue_from_json,
)
from .common import (
    ClientSetting,
    Problem,
    ProblemOutlineCache,
    ProblemOutlineGenerator,
    SolveInputsBuilder,
    feasible_outcome_details,
    log_progress,
)


_logger = logging.getLogger(__name__)


class Client:
    """Opvious API client"""

    def __init__(self, executor: Executor):
        self._executor = executor
        self._problem_outline_cache = ProblemOutlineCache(executor)

    def __repr__(self) -> str:
        fields = [
            f"executor_class={self._executor.__class__.__name__}",
            f"endpoint={json.dumps(self._executor.endpoint)}",
        ]
        return f"<opvious.Client {' '.join(fields)}>"

    @classmethod
    def default(
        cls,
        token: Union[str, bool, None] = None,
        endpoint: Optional[str] = None,
    ) -> Client:
        """
        Creates a client using the best :class:`.Executor` for the environment

        Args:
            token: API token. If absent or `True`, defaults to the
                `$OPVIOUS_TOKEN` environment variable. If `False`, no
                authentication will be set.
            endpoint: API endpoint. If absent, defaults to the
                `$OPVIOUS_ENDPOINT` environment variable, falling back to the
                cloud endpoint if neither is present.
        """
        authorization = None
        if token is True or (not token and token is not False):
            token = ClientSetting.TOKEN.read()
        if token:
            authorization = authorization_header(token.strip())
        if not endpoint:
            endpoint = ClientSetting.ENDPOINT.read()
        return Client(
            executor=default_executor(
                endpoint=endpoint,
                authorization=authorization,
            ),
        )

    @classmethod
    def from_token(cls, token: str, endpoint: Optional[str] = None) -> Client:
        return Client.default(token=token, endpoint=endpoint)

    @classmethod
    def from_environment(
        cls, env: Optional[dict[str, str]] = None, require_authenticated=False
    ) -> Client:
        token = ClientSetting.TOKEN.read(env).strip()
        if not token and require_authenticated:
            raise Exception(
                f"Missing or empty {ClientSetting.TOKEN.value} "
                "environment variable"
            )
        return Client.default(
            token=token,
            endpoint=ClientSetting.ENDPOINT.read(env),
        )

    @property
    def executor(self) -> Executor:
        """Returns the client's underlying executor"""
        return self._executor

    @property
    def authenticated(self) -> bool:
        """Returns true if the client is using a non-empty API token"""
        return self._executor.authenticated

    async def annotate_specification(
        self,
        specification: LocalSpecification,
        ignore_codes: Optional[Iterable[str]] = None,
    ) -> LocalSpecification:
        """Validates a specification, annotating it with any issues

        Args:
            specification: The specification to validate
            ignore_codes: Optional list of error codes to ignore when detecting
                issues. This can be used for example to allow unused
                definitions (`ERR_UNUSED_DEFINITION` code).
        """
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

    async def register_specification(
        self,
        specification: LocalSpecification,
        formulation_name: str,
        tag_names: Optional[Sequence[str]] = None,
    ) -> FormulationSpecification:
        """Saves a local specification within a remote formulation

        Args:
            specification: The specification to save
            formulation_name: The name of the formulation to register the
                specification in
            tag_names: Optional list of tags to assign to this specification.
                The first one, if any, will be used in the returned
                specification.

        The returned formulation can be used to queue solves for example.
        """
        await self._executor.execute_graphql_query(
            query="@RegisterSpecification",
            variables=json_dict(
                input=json_dict(
                    description=specification.description,
                    formulation_name=formulation_name,
                    sources=[s.text for s in specification.sources],
                    tag_names=tag_names,
                ),
            ),
        )
        return FormulationSpecification(
            formulation_name=formulation_name,
            tag_name=tag_names[0] if tag_names else None,
        )

    async def _prepare_problem(
        self, problem: Problem
    ) -> tuple[Json, ProblemOutline]:
        """Generates solve problem and final outline."""
        # First we fetch the outline to validate/coerce inputs later on
        if isinstance(problem.specification, FormulationSpecification):
            outline_generator, tag = await ProblemOutlineGenerator.formulation(
                executor=self._executor,
                specification=problem.specification,
            )
            formulation = json_dict(
                name=problem.specification.formulation_name,
                specification_tag_name=tag,
            )
        else:
            if isinstance(problem.specification, RemoteSpecification):
                sources = await problem.specification.fetch_sources(
                    self._executor
                )
            else:
                sources = [s.text for s in problem.specification.sources]
            formulation = json_dict(sources=sources)
            outline_generator = await ProblemOutlineGenerator.sources(
                executor=self._executor, sources=sources
            )

        # Then we apply any transformations and refresh the outline if needed
        for tf in problem.transformations or []:
            outline_generator.add_transformation(tf)
        outline, transformation_data = await outline_generator.generate()

        # Then we assemble the inputs
        builder = SolveInputsBuilder(outline=outline)
        if problem.dimensions:
            for label, dim in problem.dimensions.items():
                builder.set_dimension(label, dim)
        if problem.parameters:
            for label, param in problem.parameters.items():
                builder.set_parameter(label, param)
        inputs = builder.build()
        _logger.debug(
            "Validated inputs. [parameters=%s]",
            builder.parameter_entry_count,
        )

        # Finally we put everything together
        problem = json_dict(
            formulation=formulation,
            inputs=json_dict(
                dimensions=inputs.raw_dimensions,
                parameters=inputs.raw_parameters,
            ),
            transformations=transformation_data,
            strategy=solve_strategy_to_json(problem.strategy, outline),
            options=solve_options_to_json(problem.options),
        )
        return (problem, outline)

    async def serialize_problem(self, problem: Problem) -> Json:
        """Returns a serialized representation of the problem

        The returned JSON object is a valid `SolveCandidate` value and can be
        used to call the REST API directly.

        Args:
            problem: :class:`.Problem` instance to serialize
        """
        problem, _outline = await self._prepare_problem(problem)
        return problem

    async def summarize_problem(self, problem: Problem) -> ProblemSummary:
        """Returns summary statistics about a problem without solving it

        Args:
            problem: :class:`.Problem` instance to summarize
        """
        problem, _outline = await self._prepare_problem(problem)
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url="/summarize-problem",
            method="POST",
            json_data=json_dict(problem=problem),
        ) as res:
            return problem_summary_from_json(res.json_data())

    async def format_problem(
        self, problem: Problem, include_line_comments=False
    ) -> str:
        """Returns the problem's annotated representation in `LP format`_

        Args:
            problem: :class:`.Problem` instance to format
            include_line_comments: Include comment lines in the output. By
                default these lines are only logged as DEBUG messages.

        The LP formatted output will be fully annotated with matching keys and
        labels:

        .. code-block::

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
        problem, _outline = await self._prepare_problem(problem)
        async with self._executor.execute(
            result_type=PlainTextExecutorResult,
            url="/format-problem",
            method="POST",
            json_data=json_dict(problem=problem),
        ) as res:
            lines = []
            async for line in res.lines():
                if line.startswith("\\"):
                    _logger.debug(line[2:].strip())
                    if not include_line_comments:
                        continue
                lines.append(line)
            return "".join(lines)

    async def solve(
        self,
        problem: Problem,
        assert_feasible=False,
        prefer_streaming=True,
    ) -> Solution:
        """Solves an optimization problem remotely

        Inputs will be validated before being sent to the API for solving.

        Args:
            problem: :class:`.Problem` instance to solve
            assert_feasible: Throw if the final outcome was not feasible
            prefer_streaming: Show real time progress notifications when
                possible

        The returned solution exposes both metadata (status, objective value,
        etc.) and solution data (if the solve was feasible):

        .. code-block:: python

            solution = await client.solve(
                opvious.Problem(
                    specification=opvious.FormulationSpecification(
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
                ),
            )

            # Metadata is available on `outcome`
            print(f"Objective value: {solution.outcome.objective_value}")

            # Variable and constraint data are available via `outputs`
            optimal_allocation = solution.outputs.variable("allocation")


        See also :meth:`.Client.queue_solve` for an alternative for
        long-running solves.
        """
        problem, outline = await self._prepare_problem(problem)
        if prefer_streaming and self._executor.supports_streaming:
            problem_summary = None
            response_json = None
            async with self._executor.execute(
                result_type=JsonSeqExecutorResult,
                url="/solve",
                method="POST",
                json_data=json_dict(problem=problem),
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
                        problem_summary = problem_summary_from_json(
                            data["summary"]
                        )
                        _logger.info(
                            "Solving problem... [columns=%s, rows=%s]",
                            problem_summary.column_count,
                            problem_summary.row_count,
                        )
                    elif kind == "solving":
                        log_progress(_logger, data["progress"])
                    elif kind == "denormalized":
                        pass  # TODO: Output solution summary
                    elif kind == "solved":
                        _logger.debug("Downloaded outputs.")
                        response_json = data
                    elif kind == "error":
                        message = "Solve failed"
                        if res.trace:
                            message += f" ({res.trace})"
                        message += f": {data['error']['message']}"
                        raise Exception(message)
                    else:
                        raise Exception(
                            f"Unexpected response: {json.dumps(data)}"
                        )
            if not problem_summary or not response_json:
                raise Exception("Streaming solve terminated early")
            solution = solution_from_json(
                outline=outline,
                response_json=response_json,
                problem_summary=problem_summary,
            )
        else:
            async with self._executor.execute(
                result_type=JsonExecutorResult,
                url="/solve",
                method="POST",
                json_data=json_dict(problem=problem),
            ) as res:
                solution = solution_from_json(
                    outline=outline,
                    response_json=res.json_data(),
                )

        if isinstance(solution.outcome, FeasibleOutcome):
            details = feasible_outcome_details(solution.outcome)
            _logger.info(
                "Solve completed with status %s.%s",
                solution.status,
                f" [{details}]" if details else "",
            )
        elif assert_feasible:
            raise UnexpectedSolveOutcomeError(solution.outcome)
        else:
            _logger.info("Solve completed with status %s.", solution.status)

        return solution

    async def queue_solve(
        self, problem: Problem, annotations: Optional[list[Annotation]] = None
    ) -> Uuid:
        """Queues a solve for asynchronous processing

        Inputs will be validated locally before the request is sent to the API.
        From then on, the solve will be queued and begin solving start as soon
        as enough capacity is available.

        Args:
            problem: :class:`.Problem` instance to solve

        The returned :class:`QueuedSolve` instance can be used to:

        + track progress via :meth:`Client.poll_solve`,
        + retrieve inputs via :meth:`Client.fetch_solve`,
        + retrieve outputs via :meth:`Client.fetch_solve_outputs` (after
          successful completion).

        As a convenience, :meth:`Client.wait_for_solve_outcome` allows
        polling an solve until until it completes, backing off exponentially
        between each poll:

        .. code-block:: python

            # Queue a new Sudoku solve
            solve = await client.queue_solve(
                opvious.Problem(
                    specification=opvious.FormulationSpecification("sudoku"),
                    parameters={"hints": [(0, 0, 3), (1, 1, 5)]},
                )
            )

            # Wait for the solve to complete
            await client.wait_for_solve_outcome(solve, assert_feasible=True)

            # Fetch the solution's data
            output_data = await client.fetch_solve_outputs(solve)

            # Get a parsed variable as a dataframe
            decisions = output_data.variable("decisions")

        See also :meth:`.Client.solve` for an alternative for solving problems
        live.
        """
        if not isinstance(problem.specification, FormulationSpecification):
            raise Exception(
                "Queued solves must have a formulation as specification"
            )
        problem, _outline = await self._prepare_problem(problem)
        async with self._executor.execute(
            result_type=JsonExecutorResult,
            url="/queue-solve",
            method="POST",
            json_data=json_dict(
                problem=problem,
                annotations=encode_annotations(annotations or []),
            ),
        ) as res:
            uuid = res.json_data()["uuid"]
        _logger.info("Queued solve. [uuid=%s]", uuid)
        return uuid

    async def cancel_solve(self, uuid: Uuid) -> None:
        """Cancels a running solve

        This method will throw if the solve does not exist or is not pending
        anymore.

        Args:
            uuid: The target solve's ID
        """
        await self._executor.execute_graphql_query(
            query="@CancelQueuedSolve",
            variables=json_dict(uuid=uuid),
        )
        _logger.info("Cancelled solve. [uuid=%s]", uuid)

    async def poll_solve(
        self, uuid: Uuid
    ) -> Union[SolveNotification, SolveOutcome]:
        """Polls a solve for its outcome or latest progress notification

        Args:
            uuid: The target queued solve's UUID
        """
        data = await self._executor.execute_graphql_query(
            query="@PollQueuedSolve",
            variables=json_dict(uuid=uuid),
        )
        solve_data = data["queuedSolve"]

        error_status = solve_data["attempt"]["errorStatus"]
        if error_status:
            failure_data = solve_data["failure"]
            if failure_data:
                return failed_outcome_from_graphql(failure_data)
            else:
                return FailedOutcome(
                    error_status,
                    "The problem's inputs did not match its specification",
                )

        outcome_data = solve_data["outcome"]
        if outcome_data:
            return solve_outcome_from_graphql(outcome_data)

        edges = solve_data["notifications"]["edges"]
        return solve_notification_from_graphql(
            dequeued=bool(solve_data["dequeuedAt"]),
            data=edges[0]["node"] if edges else None,
        )

    @backoff.on_predicate(
        backoff.fibo,
        lambda ret: ret is None,
        max_value=90,
        logger=None,
    )
    async def _track_solve(self, uuid: Uuid) -> Optional[SolveOutcome]:
        ret = await self.poll_solve(uuid)
        if isinstance(ret, SolveNotification):
            if ret.dequeued:
                details: list[str] = []
                if ret.relative_gap is not None:
                    details.append(f"gap={format_percent(ret.relative_gap)}")
                if ret.cut_count is not None:
                    details.append(f"cuts={ret.cut_count}")
                if ret.lp_iteration_count is not None:
                    details.append(f"iterations={ret.lp_iteration_count}")
                _logger.info("Solve is running... [%s]", ", ".join(details))
            else:
                _logger.info("Solve is queued...")
            return None
        return ret

    async def wait_for_solve_outcome(
        self,
        uuid: Uuid,
        assert_feasible=False,
    ) -> SolveOutcome:
        """Waits for the solve to complete and returns its outcome

        This method emits real-time progress messages as INFO logs.

        Args:
            uuid: The target solve's ID
            assert_feasible: Throw if the final outcome was not feasible
        """
        _logger.debug("Tracking solve... [uuid=%s]", uuid)
        outcome = await self._track_solve(uuid)
        if not outcome:
            raise Exception("Missing outcome")
        status = solve_outcome_status(outcome)
        if isinstance(outcome, FeasibleOutcome):
            details = feasible_outcome_details(outcome)
            _logger.info(
                "Solve completed with status %s.%s",
                status,
                f" [{details}]" if details else "",
            )
        elif assert_feasible:
            raise UnexpectedSolveOutcomeError(outcome)
        else:
            _logger.info("Solve completed with status %s.", status)
        return outcome

    async def fetch_solve_inputs(self, uuid: Uuid) -> SolveInputs:
        """Retrieves a queued solve's inputs

        Args:
            uuid: The target queued solve's UUID
        """

        async def _data():
            async with self._executor.execute(
                result_type=JsonExecutorResult,
                url=f"/queued-solves/{uuid}/inputs",
            ) as res:
                return res.json_data()

        async def _outline():
            return await self._problem_outline_cache.get_solve_outline(uuid)

        data, outline = await gather(_data(), _outline())
        return SolveInputs(
            problem_outline=outline,
            raw_parameters=data["parameters"],
            raw_dimensions=data["dimensions"],
        )

    async def fetch_solve_outputs(self, uuid: Uuid) -> SolveOutputs:
        """Retrieves a successful queued solves's outputs

        This method will throw if the solve did not have a feasible solution.

        Args:
            uuid: The target queued solve's UUID
        """

        async def _data():
            async with self._executor.execute(
                result_type=JsonExecutorResult,
                url=f"/queued-solves/{uuid}/outputs",
            ) as res:
                return res.json_data()

        async def _outline():
            return await self._problem_outline_cache.get_solve_outline(uuid)

        data, outline = await gather(_data(), _outline())
        return SolveOutputs(
            problem_outline=outline,
            raw_variables=data["variables"],
            raw_constraints=data["constraints"],
        )

    async def paginate_formulation_solves(
        self,
        name: str,
        annotations: Optional[list[Annotation]] = None,
        limit: int = 25,
    ) -> AsyncIterator[QueuedSolve]:
        """Lists queued solves for a given formulation

        Args:
            name: Formulation name
            annotations: Optional annotations to filter solves by
            limit: Maximum number of solves to return

        Solves are sorted from most recently started to least.
        """
        cursor = None
        attempt_filter = json_dict(
            operation="QUEUE_SOLVE",
            annotations=encode_annotations(annotations or []),
        )

        async def _next_page() -> list[QueuedSolve]:
            nonlocal cursor
            data = await self._executor.execute_graphql_query(
                query="@PaginateFormulationAttempts",
                variables=json_dict(
                    name=name,
                    last=min(25, limit),
                    before=cursor,
                    filter=attempt_filter,
                ),
            )
            formulation = data["formulation"]
            if not formulation:
                return []
            cursor = formulation["attempts"]["pageInfo"]["startCursor"]
            solves: list[QueuedSolve] = []
            for edge in formulation["attempts"]["edges"]:
                attempt = edge["node"]
                content = attempt["content"]
                if not content:
                    continue
                solves.append(queued_solve_from_graphql(content, attempt))
            solves.reverse()
            return solves

        while limit > 0:
            solves = await _next_page()
            if not solves:
                return
            for solve in solves:
                yield solve
            limit -= len(solves)
