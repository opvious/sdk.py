import opvious.modeling as om
from typing import Optional


class TestRender:
    def test_set_cover(self):
        m = om.Model()

        m.set = om.Dimension()
        m.vertex = om.Dimension()

        m.covers = om.Parameter(m.set, m.vertex, image=om.indicator())
        m.used = om.Variable(m.set, image=om.indicator())

        @om.constrain(m)
        def all_covered():
            for v in m.vertex:
                yield om.total(m.used(g) * m.covers(g, v) for g in m.set) >= 1

        m.minimize_used = om.minimize(om.total(m.used(g) for g in m.set))

        print(om.render(m))

    def test_sudoku(self):
        m = om.Model()

        value = om.interval(1, 9, name="V")
        row = column = position = om.interval(0, 8, name="P")

        m.input = om.Parameter(row, column, value, image=om.indicator())
        m.output = om.Variable(row, column, value, image=om.indicator())

        @om.constrain(m)
        def output_matches_input():
            for i, j, v in om.cross(row, column, value):
                if m.input(i, j, v):
                    yield m.output(i, j, v) >= m.input(i, j, v)

        @om.constrain(m)
        def one_output_per_cell():
            for i, j in om.cross(row, column):
                yield om.total(m.output(i, j, v) == 1 for v in value)

        @om.constrain(m)
        def one_value_per_column():
            for j, v in om.cross(column, value):
                yield om.total(m.output(i, j, v) == 1 for i in row)

        @om.constrain(m)
        def one_value_per_row():
            for i, v in om.cross(row, value):
                yield om.total(m.output(i, j, v) == 1 for j in column)

        @om.constrain(m)
        def one_value_per_box():
            for v, b in om.cross(value, position):
                yield om.total(
                    m.output(3 * (b // 3) + c // 3, 3 * (b % 3) + c % 3, v)
                    == 1
                    for c in position
                )

        print(om.render(m))


class SetCover(om.Model):
    sets = om.Dimension()
    vertices = om.Dimension()
    covers = om.Parameter(sets, vertices, image=om.indicator())
    used = om.Variable(sets, image=om.indicator())

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
    horizon = om.Parameter()
    steps = om.interval(1, horizon)

    holding_cost = om.Parameter(steps)
    setup_cost = om.Parameter(steps)
    demand = om.Parameter(steps, image=om.non_negative())

    production = om.Variable(steps, image=om.non_negative())
    inventory = om.Variable(steps, image=om.non_negative())
    is_producing, is_producing_activates, _ = om.patterns.activation(
        variable=production,
        upper_bound=om.total(demand(t) for t in steps),
        omit_deactivation=True,
    )

    @om.objective
    def minimize_total_cost(self):
        return om.total(
            self.holding_cost(t) * self.inventory(t)
            + self.setup_cost(t) * self.is_producing(t)
            for t in self.steps
        )

    @om.constraint
    def inventory_propagates(self):
        for t in self.steps:
            prev = om.switch((t > 0, self.inventory(t - 1)), 0)
            diff = self.production(t) - self.demand(t)
            yield self.inventory(t) == prev + diff


class GroupExpenses(om.Model):
    friends = om.Dimension()
    transactions = om.Dimension()

    paid = om.Parameter(transactions, friends)
    share = om.Parameter(transactions, friends)
    floor = om.Parameter()

    max_transfer_count = om.Variable(image=om.natural())

    transferred = om.Variable(
        {"sender": friends, "recipient": friends},
        image=om.non_negative_real(),
    )

    @om.alias(transactions, name="tc")
    def transaction_cost(self, t):
        return om.total(self.paid(t, f) for f in self.friends)

    @property
    @om.alias(name="tp")
    def total_paid(self):
        return om.total(self.transaction_cost(t) for t in self.transactions)

    is_transferring, is_transferring_activation = activation_variable(
        variable=m.transferred, upper_bound=self.total_paid
    )

    def relative_share(self, t, f):
        return self.share(t, f) / om.total(
            self.share(t, f) for f in self.friends
        )

    @om.constraint
    def payments_are_settled(self):
        for f in self.friends:
            received = om.total(self.transferred(s, f) for s in self.friends)
            sent = om.total(self.transferred(f, r) for r in self.friends)
            owed = om.total(
                self.paid(t, f)
                - transaction_cost(t) * self.relative_share(t, f)
                for t in self.transactions
            )
            yield received - sent == owed

    @om.constraint
    def transfer_count_is_below_max(self):
        for s in self.friends:
            transfer_count = om.total(
                is_transferring(s, r) for r in self.friends
            )
            yield transfer_count <= m.max_transfer_count

    @om.objective
    def minimize_transfer_count(self):
        return self.max_transfer_count

    @om.objective
    def minimize_total_transferred(self):
        return om.total(
            m.transferred(s, r)
            for s, r in om.cross(self.friends, self.friends)
        )


class Sudoku(om.Model):
    values = om.interval(1, 9, name="V")
    rows = columns = positions = om.interval(0, 8, name="P")

    input = om.Parameter(rows, columns, values, image=om.indicator())
    output = om.Variable(rows, columns, values, image=om.indicator())

    @om.constraint
    def output_matches_input(self):
        for i, j, v in om.cross(self.rows, self.columns, self.values):
            if self.input(i, j, v):
                yield self.output(i, j, v) >= self.input(i, j, v)

    @om.constraint
    def one_output_per_cell(self):
        for i, j in om.cross(self.rows, self.columns):
            yield om.total(self.output(i, j, v) == 1 for v in self.values)

    @om.constraint
    def one_value_per_column(self):
        for j, v in om.cross(self.columns, self.values):
            yield om.total(self.output(i, j, v) == 1 for i in self.rows)

    @om.constraint
    def one_value_per_row(self):
        for i, v in om.cross(self.rows, self.values):
            yield om.total(self.output(i, j, v) == 1 for j in self.columns)

    @om.constraint
    def one_value_per_box(self):
        for v, b in om.cross(self.values, self.positions):
            yield om.total(
                self.output(3 * (b // 3) + c // 3, 3 * (b % 3) + c % 3, v) == 1
                for c in self.positions
            )


def activation(
    variable: om.VariableDefinition,
    upper_bound: Union[float, bool] = None,
    lower_bound: Union[float, bool] = None,
) -> Tuple[om.Variable, Optional[om.Constraint], Optional[om.Constraint]]:
    m = variable.model

    activation = om.define(
        m,
        f"{variable.label}_is_active",
        om.Variable(*variable.sources, image=om.indicator()),
    )

    if not upper_bound is False:
        # TODO: Infer bound when `True`

        @om.constrain(m, label=f"{variable.label}_activates")
        def activates():
            for t in om.cross(*variable.sources):
                yield variable(*t) <= upper_bound * activation(*t)

    if not lower_bound is False:
        # TODO: Infer bound when `True`

        @om.constrain(m, label=f"{variable.label}_deactivates")
        def deactivates():
            for t in om.cross(*variable.sources):
                yield variable(*t) >= lower_bound * activation(*t)

    return (activation,)
