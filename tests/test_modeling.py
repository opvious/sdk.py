import opvious
import pytest


om = opvious.modeling
client = opvious.Client.from_environment()


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

        self.holding_cost = om.Parameter.non_negative(self.steps, name="c")
        self.setup_cost = om.Parameter.non_negative(self.steps)
        self.demand = om.Parameter.non_negative(self.steps)

        self.production = om.Variable.non_negative(self.steps)
        self.production_indicator = om.fragments.ActivationIndicator(
            variable=self.production,
            upper_bound=om.total(self.demand(t) for t in self.steps),
        )
        self.inventory = om.Variable.non_negative(self.steps)

    @property
    @om.alias("T")
    def steps(self) -> om.Quantified[om.Quantifier]:
        return om.interval(1, self.horizon())

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
            variable=self.transferred,
            upper_bound=self.overall_cost(),
            lower_bound=False,
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
        for s, r in om.cross(self.friends, self.friends):
            floor = self.floor() * self.tranferred_indicator(s, r)
            yield self.transferred(s, r) >= floor

    @om.objective
    def minimize_transfer_count(self) -> om.Expression:
        return self.max_transfer_count()

    @om.objective
    def minimize_total_transferred(self) -> om.Expression:
        return om.total(
            self.transferred(s, r)
            for s, r in om.cross(self.friends, self.friends)
        )


class Sudoku(om.Model):
    _qualifiers = ["row", "column", "value"]

    def __init__(self) -> None:
        self.input = om.Parameter.indicator(
            (self.grid, self.values),
            qualifiers=self._qualifiers,
        )
        self.output = om.Variable.indicator(
            (self.grid, self.values),
            qualifiers=self._qualifiers,
        )

    @property
    @om.alias("V")
    def values(self) -> om.Quantified[om.Quantifier]:
        return om.interval(1, 9)

    @property
    @om.alias("P")
    def positions(self) -> om.Quantified[om.Quantifier]:
        return om.interval(0, 8)

    @property
    @om.alias("G")
    def grid(self) -> om.Quantification:
        return om.cross(self.positions, self.positions)

    @om.constraint(qualifiers=_qualifiers)
    def output_matches_input(self):
        for i, j, v in om.cross(self.positions, self.positions, self.values):
            if self.input(i, j, v):
                yield self.output(i, j, v) >= self.input(i, j, v)

    @om.constraint
    def one_output_per_cell(self):
        for i, j in self.grid:
            yield om.total(self.output(i, j, v) == 1 for v in self.values)

    @om.constraint
    def one_value_per_column(self):
        for j, v in om.cross(self.positions, self.values):
            yield om.total(self.output(i, j, v) == 1 for i in self.positions)

    @om.constraint
    def one_value_per_row(self):
        for i, v in om.cross(self.positions, self.values):
            yield om.total(self.output(i, j, v) == 1 for j in self.positions)

    @om.constraint
    def one_value_per_box(self):
        for v, b in om.cross(self.values, self.positions):
            yield om.total(
                self.output(3 * (b // 3) + c // 3, 3 * (b % 3) + c % 3, v) == 1
                for c in self.positions
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


@pytest.mark.skipif(
    not client.authenticated, reason="No access token detected"
)
class TestModeling:
    _models = [
        SetCover(),
        LotSizing(),
        GroupExpenses(),
        Sudoku(),
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
            holidays = om.fragments.MaskedSubset(days, alias_name="H")
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
