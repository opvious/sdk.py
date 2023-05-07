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


def lot_sizing():
    m = om.Model()

    m.horizon = om.Parameter()
    m.steps = om.Interval(1, horizon)

    m.holding_cost = om.Parameter(steps)
    m.setup_cost = om.Parameter(steps)
    m.demand = om.Parameter(steps, image=om.non_negative())

    m.production = om.Variable(steps, image=om.non_negative())
    m.inventory = om.Variable(steps, image=om.non_negative())

    is_producing = activation_variable(
        variable=m.production, upper_bound=om.total(m.demand(t) for t in steps)
    )

    m.minimize_total_cost = om.minimize(
        om.total(
            m.holding_cost(t) * m.inventory(t)
            + m.setup_cost(t) * is_producing(t)
            for t in steps
        )
    )

    @om.constrain(m)
    def inventory_propagates():
        for t in steps:
            prev = om.switch((t > 0, inventory(t - 1)), 0)
            yield m.inventory(t) == prev + m.production(t) - m.demand(t)


def activation_variable(
    variable: om.VariableDefinition,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None
    # TODO: Projection
) -> om.VariableDefinition:
    m = variable.model

    activation = om.define(
        m,
        f"{variable.label}_is_active",
        om.Variable(*variable.sources, image=om.indicator()),
    )

    if not upper_bound is False:
        # TODO: Infer bound when `None`

        @om.constrain(m, label=f"{variable.label}_activates")
        def activates():
            for t in om.cross(*variable.sources):
                yield variable(*t) <= upper_bound * activation(*t)

    if not lower_bound is False:
        # TODO: Infer bound when `None`

        @om.constrain(m, label=f"{variable.label}_deactivates")
        def deactivates():
            for t in om.cross(*variable.sources):
                yield variable(*t) >= activation(*t)

    return activation


def product_allocation():
    m = Model()

    m.product = om.Dimension()
    m.size = om.Dimension()
    m.location = om.Dimension()
    m.demand_tier = om.Dimension(name="T")

    m.supply = (om.Parameter(m.product, m.size),)
    m.min_allocation = (om.Parameter(m.product, name="a^{min}"),)
    m.max_total_allocation = (om.Parameter(name="a^{max}"),)
    m.diversity = om.Parameter(m.product)
    m.demand = om.Parameter(m.location, m.product, m.size, m.demand_tier)
    m.demand_value = om.Parameter(m.demand_tier)

    m.allocation = om.Variable(m.location, m.product, m.size, m.demand_tier)

    is_size_allocated = activation_variable(
        variable=m.allocation,
        projection=(True, True, True, False),
        upper_bound=supply,
    )
    is_product_allocated = activation_variable(
        variable=is_size_allocated,
        projection=(True, True, False),
        # Upper bound can be omitted since the variable is bounded (by 1)
    )

    def tier_allocation(l, p, s):
        return total(m.allocation(l, p, s, t) for t in m.demand_tier)

    @enforce(m)
    def at_most_demand():
        for t, p, s, l in cross(m.demand_tier, m.product, m.size, m.location):
            return m.allocation(l, p, s, t) <= m.demand(l, p, s, t)

    @enforce(m)
    def at_most_supply():
        for p, s in cross(m.product, m.size):
            alloc = total(tier_allocation(l, p, s) for l in m.location)
            return alloc <= m.supply(p, s)

    @enforce(m)
    def at_most_max_total_allocation():
        alloc = total(
            tier_allocation(l, p, s)
            for l, p, s in cross(m.location, m.product, m.size)
        )
        return alloc <= m.max_total_allocation

    @enforce(m)
    def at_least_min_allocation():
        for p, l in cross(m.product, m.location):
            alloc = total(tier_allocation(l, p, s) for s in m.size)
            return alloc >= a.min_allocation(p) * is_product_allocated(l, p)

    @enforce(m)
    def at_least_min_diversity():
        for p, l in cross(m.product, m.location):
            avail = total(m.is_size_allocated(l, p, s) for s in m.size)
            return alloc >= a.diversity(p) * m.is_product_allocated(l, p)

    @maximize(m)
    def maximize_value():
        return total(
            m.demand_value(t) * m.allocation(l, p, s, t)
            for l, p, s, t in cross(
                m.location, m.product, m.size, m.demand_tier
            )
        )

    return m


def sudoku():
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
            i = 3 * (b // 3) + c // 3
            j = 3 * (b % 3) + c % 3
            yield om.total(m.output(i, j, v) == 1 for c in position)

    return m


def group_expenses():
    m = om.Model()

    m.friend = om.Dimension()
    m.transaction = om.Dimension()

    m.paid = om.Parameter(m.transaction, m.friend)
    m.share = om.Parameter(m.transaction, m.friend)
    m.floor = om.Parameter()

    m.transferred = om.Variable(
        {"sender": m.friend, "recipient": m.friend},
        image=om.non_negative_real(),
    )

    @om.alias(m, "tc")
    def transaction_cost(t) -> om.Expression:
        return om.total(m.paid(t, f) for f in m.friend)

    @om.alias(m, "tp")
    def total_paid() -> om.Expression:
        return om.total(transaction_cost(t) for t in m.transaction)

    is_transferring = activation_variable(
        variable=m.transferred, upper_bound=total_paid
    )

    pairs = om.alias(m, "S", om.cross(m.friend, m.friend))

    @om.alias(m, "S")
    def pairs() -> om.Space:
        return om.cross(m.friend, m.friend)

    m.max_transfer_count = om.Variable(image=om.natural(om.count(m.friend)))

    def relative_share(t, f):
        return m.share(t, f) / om.total(m.share(t, f) for f in m.friend)

    @om.constrain(m)
    def payments_are_settled():
        for f in friend:
            received = om.total(m.transferred(s, f) for s in m.friend)
            sent = om.total(m.transferred(f, r) for r in m.friend)
            owed = om.total(
                m.paid(t, f) - transaction_cost(t) * relative_share(t, f)
                for t in transaction
            )
            yield received - sent == owed

    @om.constrain(m)
    def transfer_count_is_below_max():
        for s in friend:
            transfer_count = om.total(is_transferring(s, r) for r in friend)
            yield transfer_count <= m.max_transfer_count

    m.minimize_transfer_count = m.minimize(m.max_transfer_count)

    m.minimize_total_transferred = m.minimize(
        om.total(m.transferred(s, r) for s, r in om.cross(m.friend, m.friend))
    )

    return m
