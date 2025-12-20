from opvious.data.outlines import ProblemOutline, TensorOutline
from opvious.data.solves import SolveOutputs


class TestSolveOutputs:
    def test_variable_empty(self):
        outputs = SolveOutputs(
            problem_outline=ProblemOutline(
                objectives={},
                dimensions={},
                parameters={},
                variables={
                    "foo": TensorOutline(
                        label="foo",
                        lower_bound=None,
                        upper_bound=None,
                        is_integral=False,
                        bindings=[],
                        derivation_kind=None,
                    ),
                },
                constraints={},
            ),
            raw_variables=[{"label": "foo", "entries": []}],
            raw_constraints=[],
        )
        df = outputs.variable("foo")
        assert len(df["value"]) == 0
