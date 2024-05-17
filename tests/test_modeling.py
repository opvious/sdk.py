import opvious
import pytest


om = opvious.modeling
client = opvious.Client.default()


class SetCover(om.Model):
    sets = om.Dimension()
    vertices = om.Dimension()
    covers = om.Parameter.indicator(sets, vertices)
    used = om.Variable.indicator(sets)

    @om.constraint
    def all_covered(self):
        for v in self.vertices:
            count = om.total(
                self.used(s) * self.covers(s, v) for s in self.sets
            )
            yield count >= 1

    @om.objective
    def minimize_sets(self):
        return om.total(self.used(s) for s in self.sets)


class LotSizing(om.Model):
    def __init__(self) -> None:
        self.horizon = om.Parameter.natural()
        self.steps = om.interval(1, self.horizon(), name="T")

        self.holding_cost = om.Parameter.non_negative(self.steps, name="c")
        self.setup_cost = om.Parameter.non_negative(self.steps)
        self.demand = om.Parameter.non_negative(self.steps)

        self.production = om.Variable.non_negative(
            self.steps, upper_bound=self.demand.total(absolute=True)
        )
        self.production_indicator = om.fragments.ActivationIndicator(
            tensor=self.production,
        )
        self.inventory = om.Variable.non_negative(self.steps)

    @om.objective()
    def minimize_total_cost(self) -> om.Expression:
        return om.total(
            self.holding_cost(t) * self.inventory(t)
            + self.setup_cost(t) * self.production_indicator(t)
            for t in self.steps
        )

    @om.constraint()
    def inventory_propagates(self) -> om.Quantified[om.Predicate]:
        for t in self.steps:
            prev = om.switch((t > 0, self.inventory(t - 1)), 0)
            diff = self.production(t) - self.demand(t)
            yield self.inventory(t) == prev + diff


class GroupExpenses(om.Model):
    def __init__(self) -> None:
        self.friends = om.Dimension()
        self.transactions = om.Dimension()

        self.paid = om.Parameter.continuous(self.transactions, self.friends)
        self.share = om.Parameter.non_negative(self.transactions, self.friends)
        self.floor = om.Parameter.non_negative(name="t^\\mathrm{min}")

        self.max_transfer_count = om.Variable.natural(
            upper_bound=om.size(self.friends)
        )
        self.transferred = om.Variable.non_negative(
            (self.friends, self.friends),
            qualifiers=["sender", "recipient"],
        )
        self.tranferred_indicator = om.fragments.ActivationIndicator(
            tensor=self.transferred,
            upper_bound=self.overall_cost(),
        )

    @om.alias("tc")
    def transaction_cost(self, t) -> om.Expression:
        return om.total(self.paid(t, f) for f in self.friends)

    @om.alias("op")
    def overall_cost(self) -> om.Expression:
        return om.total(self.transaction_cost(t) for t in self.transactions)

    @om.alias("rs")
    def relative_share(self, t, f) -> om.Expression:
        total_share = om.total(self.share(t, f) for f in self.friends)
        return self.share(t, f) / total_share

    @om.constraint
    def debts_are_settled(self) -> om.Quantified[om.Predicate]:
        for f in self.friends:
            received = om.total(self.transferred(s, f) for s in self.friends)
            sent = om.total(self.transferred(f, r) for r in self.friends)
            owed = om.total(
                self.paid(t, f)
                - self.transaction_cost(t) * self.relative_share(t, f)
                for t in self.transactions
            )
            yield received - sent == owed

    @om.constraint
    def transfer_count_is_below_max(self) -> om.Quantified[om.Predicate]:
        for s in self.friends:
            transfer_count = om.total(
                self.tranferred_indicator(s, r) for r in self.friends
            )
            yield transfer_count <= self.max_transfer_count()

    @om.constraint
    def transfer_is_above_floor(self) -> om.Quantified[om.Predicate]:
        for s, r in self.friends * self.friends:
            floor = self.floor() * self.tranferred_indicator(s, r)
            yield self.transferred(s, r) >= floor

    @om.objective
    def minimize_transfer_count(self) -> om.Expression:
        return self.max_transfer_count()

    @om.objective
    def minimize_total_transferred(self) -> om.Expression:
        return om.total(
            self.transferred(s, r) for s, r in self.friends * self.friends
        )


class Sudoku(om.Model):
    _qualifiers = ["row", "column", "value"]
    values = om.interval(1, 9, name="V")
    positions = om.interval(0, 8, name="P")

    def __init__(self) -> None:
        self.input = om.Parameter.indicator(
            self.grid * self.values,
            qualifiers=self._qualifiers,
        )
        self.output = om.Variable.indicator(
            self.grid * self.values,
            qualifiers=self._qualifiers,
        )

    @property
    @om.alias("G")
    def grid(self) -> om.Quantification:
        return self.positions * self.positions

    @om.constraint(qualifiers=_qualifiers)
    def output_matches_input(self):
        for i, j, v in self.grid * self.values:
            if self.input(i, j, v):
                yield self.output(i, j, v) >= self.input(i, j, v)

    @om.constraint
    def one_output_per_cell(self):
        for i, j in self.grid:
            yield om.total(self.output(i, j, v) == 1 for v in self.values)

    @om.constraint
    def one_value_per_column(self):
        for j, v in self.positions * self.values:
            yield om.total(self.output(i, j, v) == 1 for i in self.positions)

    @om.constraint
    def one_value_per_row(self):
        for i, v in self.positions * self.values:
            yield om.total(self.output(i, j, v) == 1 for j in self.positions)

    @om.constraint
    def one_value_per_box(self):
        for v, b in self.values * self.positions:
            yield om.total(
                self.output(3 * (b // 3) + c // 3, 3 * (b % 3) + c % 3, v) == 1
                for c in self.positions
            )


class BinPacking(om.Model):
    items = om.Dimension()
    bins = om.interval(1, om.size(items))

    weight = om.Parameter.non_negative(items)
    bin_max_weight = om.Parameter.non_negative()

    assigned = om.Variable.indicator(bins, items)
    used = om.fragments.ActivationIndicator(assigned, projection=1)

    @om.objective
    def minimize_bins_used(self):
        return om.total(self.used(b) for b in self.bins)

    @om.constraint
    def each_item_is_assigned_once(self):
        for i in self.items:
            yield om.total(self.assigned(b, i) for b in self.bins) == 1

    @om.constraint
    def bins_are_below_max_weight(self):
        for b in self.bins:
            yield self._bin_weight(b) <= self.bin_max_weight()

    def _bin_weight(self, b):
        return om.total(
            self.weight(i) * self.assigned(b, i) for i in self.items
        )


class InvalidSetCoverModel(om.Model):
    sets = om.Dimension()
    vertices = om.Dimension()
    covers = om.Parameter.indicator(sets, vertices)
    used = om.Variable.indicator(sets)

    @om.constraint
    def all_covered(self):
        for v in self.vertices:
            count = om.total(
                self.used(s) * self.covers(v, s)  # Bad subscript order
                for s in self.sets
            )
            yield count >= 1


class JobShopScheduling(om.Model):
    tasks = om.Dimension()
    duration = om.Parameter.natural(tasks)
    machine = om.Parameter.natural(tasks)
    dependency = om.Parameter.indicator(
        (tasks, tasks), qualifiers=["child", "parent"]
    )
    task_start = om.Variable.natural(tasks)
    horizon = om.Variable.natural()

    @property
    def competing_tasks(self):
        for t1, t2 in self.tasks * self.tasks:
            if t1 != t2 and self.machine(t1) == self.machine(t2):
                yield t1, t2

    def task_end(self, t):
        return self.task_start(t) + self.duration(t)

    @om.constraint
    def all_tasks_end_within_horizon(self):
        for t in self.tasks:
            yield self.task_end(t) <= self.horizon()

    @om.constraint
    def child_starts_after_parent_ends(self):
        for c, p in self.tasks * self.tasks:
            if self.dependency(c, p):
                yield self.task_start(c) >= self.task_end(p)

    @om.fragments.activation_variable(
        lambda init, self: init(
            self.competing_tasks,
            negate=True,
            upper_bound=self.duration.total(),
        )
    )
    def must_start_after(self, t1, t2):
        return self.task_end(t2) - self.task_start(t1)

    @om.fragments.activation_variable(
        lambda init, self: init(
            self.competing_tasks,
            negate=True,
            upper_bound=self.duration.total(),
        )
    )
    def must_end_before(self, t1, t2):
        return self.task_end(t1) - self.task_start(t2)

    @om.constraint
    def one_active_task_per_machine(self):
        for t1, t2 in self.competing_tasks:
            yield self.must_end_before(t1, t2) + self.must_start_after(
                t1, t2
            ) >= 1

    @om.objective
    def minimize_horizon(self):
        return self.horizon()


@pytest.mark.skipif(
    not client.authenticated, reason="No access token detected"
)
class TestModeling:
    _models = [
        SetCover(),
        LotSizing(),
        GroupExpenses(),
        Sudoku(),
        BinPacking(),
        JobShopScheduling(),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", _models)
    async def test_specification(self, model):
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_definition_counts(self):
        model = LotSizing()
        counts = model.definition_counts()
        assert counts["PARAMETER"].iloc[0] == 4

    @pytest.mark.asyncio
    async def test_global_name_collision(self):
        class _Model(om.Model):
            count = om.Variable.natural()
            cost = om.Variable.natural()
            step_count = om.Variable.natural()
            step_cost = om.Variable.non_negative()
            step_color = om.Variable.non_negative()

            @om.objective
            def minimize_cost(self):
                return (
                    self.count()
                    + self.cost()
                    + self.step_count()
                    + self.step_cost()
                    + self.step_color()
                )

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0
        text = spec.sources[0].text
        assert r"\chi^\mathrm{step \prime}" in text
        assert r"\chi^\mathrm{step \prime \prime}" in text
        assert r"\chi'" in text

    @pytest.mark.asyncio
    async def test_annotate_markdown_repr(self):
        model = InvalidSetCoverModel()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 2

    @pytest.mark.asyncio
    async def test_model_space_alias(self):
        class _Model(om.Model):
            days = om.Dimension()
            steps = om.Dimension()
            target = om.Variable.continuous(days, steps)

            @property
            @om.alias("T")
            def times(self):
                return om.cross(self.days, self.steps)

            @om.objective
            def minimize_target(self):
                return om.total(self.target(d, s) for d, s in self.times)

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0
        assert "(d, s) \\in T" in spec.sources[0].text

    @pytest.mark.asyncio
    async def test_masked_subset(self):
        class _Model(om.Model):
            days = om.Dimension()
            holidays = om.fragments.MaskedSubspace(days, alias_name="H")
            target = om.Variable.discrete(holidays)

            @om.constraint
            def nothing_on_holidays(self):
                yield om.total(self.target(d) for d in self.holidays) == 0

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert (
            r"H \doteq \{ d \in D \mid m^\mathrm{holidays}_{d} \neq 0 \}"
            in text
        )
        assert r"\tau \in \mathbb{Z}^{H}" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_fragment_alias_call(self):
        class _Fragment(om.ModelFragment):
            def __init__(self, dim):
                self._dimension = dim

            @om.alias("C")
            def at_least(self, val):
                for d in self._dimension:
                    if d >= val:
                        yield d

        class _Model(om.Model):
            values = om.Dimension(is_numeric=True)
            above = _Fragment(values)
            target = om.Variable.continuous(values)

            @om.constraint
            def zero_above_two(self):
                for v in self.above.at_least(2):
                    yield self.target(v) == 0

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0
        text = spec.sources[0].text
        assert r"\forall c \in C_{2}, \tau_{c} = 0" in text

    @pytest.mark.asyncio
    async def test_dependency_prefix(self):
        class _Nested(om.Model):
            def __init__(self, prefix: str) -> None:
                super().__init__(prefix=[prefix])
                self.target = om.Variable.continuous()

        class _Model(om.Model):
            def __init__(self) -> None:
                self.bar = _Nested("bar")
                self.foo = _Nested("foo")
                super().__init__([self.bar, self.foo])

            @om.objective
            def maximize_targets(self):
                return self.bar.target() + self.foo.target()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0
        text = spec.sources[0].text
        assert r"\max \tau^\mathrm{bar} + \tau^\mathrm{foo}" in text

    @pytest.mark.asyncio
    async def test_derived_variable(self):
        class _Model(om.Model):
            values = om.Dimension()
            colors = om.Dimension()
            target = om.Variable.continuous(values, colors)

            @om.fragments.derived_variable(colors, name="t^c")
            def target_by_color(self, c) -> om.Expression:
                return om.total(self.target(v, c) for v in self.values)

            @om.objective
            def maximize_target(self) -> om.Expression:
                return om.total(self.target_by_color(c) for c in self.colors)

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0
        text = spec.sources[0].text
        assert r"\forall c \in C, t^c_{c} = \sum_{v \in V} \tau_{v,c}" in text

    @pytest.mark.asyncio
    async def test_derived_variable_alias_quantifiable(self):
        class _Colors(om.Model):
            available = om.Dimension()

            @property
            @om.alias("P")
            def available_pairs(self):
                for c1, c2 in om.cross(self.available, self.available):
                    if c1 != c2:
                        yield c1, c2

        colors = _Colors()

        class _Model(om.Model):
            cost = om.Parameter.non_negative(colors.available)
            target = om.Variable.natural(colors.available_pairs)

            def __init__(self) -> None:
                super().__init__([colors])

            @om.fragments.derived_variable(colors.available_pairs)
            def cost_by_pair(self, c1, c2) -> om.Expression:
                return self.cost(c1) * self.cost(c2) * self.target(c1, c2)

            @om.objective
            def minimize_cost(self) -> om.Expression:
                return om.total(
                    self.cost_by_pair(*tp) for tp in colors.available_pairs
                )

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_magnitude(self):
        class _Model(om.Model):
            points = om.Dimension()
            offset = om.Variable.continuous(points)

            @om.fragments.magnitude_variable(points, projection=0, name="\\mu")
            def magnitude(self, p) -> om.Expression:
                return 2 * self.offset(p)

            @om.objective
            def minimize_magnitude(self) -> om.Expression:
                return self.magnitude()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\forall p \in P, {-\mu} \leq 2 \omicron_{p}" in text
        assert r"\forall p \in P, \mu \geq 2 \omicron_{p}" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_expression_alias(self):
        class _Base(om.Model):
            horizon = om.Parameter.natural()
            doctors = om.Dimension(name="I")
            days = om.interval(1, horizon(), name="T")
            shifts = om.interval(1, 3, name="K")

            assigned = om.Variable.indicator(doctors, days, shifts)

            @om.alias("\\beta")
            def started_shift(self, i, t, k):
                return self.assigned(i, t, k) - om.switch(
                    (t > 1, self.assigned(i, t - 1, k)), 0
                )

            @property
            def assignment_space(self):
                return om.cross(self.assigned.quantifiables())

        base = _Base()

        class SwitchOnShiftStart(om.Model):
            def __init__(self):
                super().__init__([base])
                self.switched = om.Variable.indicator(base.doctors, base.days)

            @om.constraint
            def shift_start_forces_switch(self):
                for i, t, k in base.assignment_space:
                    yield self.switched(i, t) >= base.started_shift(i, t, k)

        model = SwitchOnShiftStart()
        spec = await client.annotate_specification(model.specification())
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_activation_variable_lower_bound(self):
        class _Model(om.Model):
            products = om.Dimension()
            locations = om.Dimension()
            sizes = om.Dimension()
            allocation = om.Variable.natural(locations, products, sizes)
            size_allocated = om.fragments.ActivationVariable(
                allocation,
                projection=0b101,
                upper_bound=False,
                lower_bound=1,
            )

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\sum_{p \in P} \alpha_{l,p,s}" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_magnitude_variable_projection(self):
        class _Model(om.Model):
            members = om.Dimension()
            max_transfer = om.Parameter.non_negative()
            transfer = om.Variable.non_negative(
                members, members, upper_bound=max_transfer()
            )
            is_transferring = om.fragments.ActivationVariable(transfer)

            @om.fragments.magnitude_variable(
                members, projection=0, lower_bound=False
            )
            def max_transfer_count(self, s):
                return om.total(
                    self.is_transferring(s, r) for r in self.members
                )

            @om.objective
            def minimize_max_transfer_count(self):
                return self.max_transfer_count()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert "UpperBounds" in text
        assert "LowerBounds" not in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_abs(self):
        class _Model(om.Model):
            members = om.Dimension()
            payment = om.Parameter.continuous(members)

            def __init__(self):
                super().__init__()
                self.transfer = om.Variable.non_negative(
                    upper_bound=om.total(
                        abs(self.payment(m)) for m in self.members
                    )
                )

            @om.objective
            def minimize_transfer(self):
                return self.transfer()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\lvert p_{m} \rvert" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_alias_explicit_quantifiable(self):
        class _Model(om.Model):
            horizon = om.Parameter.natural()
            days = om.interval(1, horizon(), name="D")
            shifts = om.Dimension()
            schedule = om.Variable.indicator(days, shifts)

            @om.alias(r"\lambda", days)
            def unscheduled(self, d):
                return 1 - om.total(self.schedule(d, s) for s in self.shifts)

            @om.constraint
            def at_most_five_shifts_per_week(self):
                for d in self.days:
                    if d < self.horizon() - 5:
                        yield om.total(
                            self.unscheduled(f) for f in om.interval(d, d + 6)
                        ) >= 2

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\sum_{x \in \{ d \ldots d + 6 \}}" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_sub_precedence(self):
        class _Model(om.Model):
            actual = om.Variable.indicator()

            def opposite(self):
                return 1 - self.actual()

            @om.objective
            def minimize(self):
                return self.actual() - self.opposite()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\alpha - \left(1 - \alpha\right)" in text
        assert spec.annotation.issue_count == 0

    @pytest.mark.asyncio
    async def test_piecewise_linear(self):
        class _Model(om.Model):
            products = om.Dimension()
            builds = om.Variable.natural(products)
            build_cost = om.fragments.PiecewiseLinear(
                builds, 2, assume_convex=True, component_name="c^{p%}"
            )

            @om.objective
            def minimize_cost(self):
                return self.build_cost.total()

        model = _Model()
        spec = await client.annotate_specification(model.specification())
        text = spec.sources[0].text
        assert r"\forall p \in P, c^{p0}_{p} + c^{p1}_{p} = \beta_{p}" in text
        assert spec.annotation.issue_count == 0
