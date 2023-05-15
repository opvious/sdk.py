import opvious.modeling as om


class TestRender:
    def test_set_cover(self):
        m = SetCover()
        print(m.render_specification())

    def test_sudoku(self):
        m = Sudoku()
        print(m.render_specification())

    def test_lot_sizing(self):
        m = LotSizing()
        print(m.render_specification())

    def test_group_expenses(self):
        m = GroupExpenses()
        print(m.render_specification())


class SetCover(om.Model):
    sets = om.Dimension()
    vertices = om.Dimension()
    covers = om.Parameter(sets, vertices, image=om.Image.indicator())
    used = om.Variable(sets, image=om.Image.indicator())

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
        self.horizon = om.Parameter(image=om.Image.natural())

        self.holding_cost = om.Parameter(self.steps, name="c")
        self.setup_cost = om.Parameter(self.steps)
        self.demand = om.Parameter(self.steps, image=om.Image.non_negative())

        self.production = om.Variable(
            self.steps, image=om.Image.non_negative()
        )
        self.production_indicator = om.ActivationIndicator.fragment(
            variable=self.production,
            upper_bound=om.total(self.demand(t) for t in self.steps),
        )
        self.inventory = om.Variable(self.steps, image=om.Image.non_negative())

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

        self.paid = om.Parameter(self.transactions, self.friends)
        self.share = om.Parameter(self.transactions, self.friends)
        self.floor = om.Parameter()

        self.max_transfer_count = om.Variable(image=om.Image.natural())
        self.transferred = om.Variable(
            (self.friends, self.friends),
            image=om.Image.non_negative(),
            qualifiers=["sender", "recipient"],
        )
        self.tranferred_indicator = om.ActivationIndicator.fragment(
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

    @om.constraint()
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

    @om.constraint()
    def transfer_count_is_below_max(self) -> om.Quantified[om.Predicate]:
        for s in self.friends:
            transfer_count = om.total(
                self.tranferred_indicator(s, r) for r in self.friends
            )
            yield transfer_count <= self.max_transfer_count()

    @om.objective()
    def minimize_transfer_count(self) -> om.Expression:
        return self.max_transfer_count()

    @om.objective()
    def minimize_total_transferred(self) -> om.Expression:
        return om.total(
            self.transferred(s, r)
            for s, r in om.cross(self.friends, self.friends)
        )


class Sudoku(om.Model):
    _qualifiers = ["row", "column", "value"]

    def __init__(self) -> None:
        self.input = om.Parameter(
            (self.positions, self.positions, self.values),
            image=om.Image.indicator(),
            qualifiers=self._qualifiers,
        )
        self.output = om.Variable(
            (self.positions, self.positions, self.values),
            image=om.Image.indicator(),
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

    @om.constraint(qualifiers=_qualifiers)
    def output_matches_input(self):
        for i, j, v in om.cross(self.positions, self.positions, self.values):
            if self.input(i, j, v):
                yield self.output(i, j, v) >= self.input(i, j, v)

    @om.constraint
    def one_output_per_cell(self):
        for i, j in om.cross(self.positions, self.positions):
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
