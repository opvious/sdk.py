from opvious.modeling import Model
from opvious.notebooks import load_notebook_models


class TestNotebooks:
    def test_load_notebook_models(self):
        ns = load_notebook_models(
            "notebooks/set-cover.ipynb",
            root=__file__,
        )
        spec = ns.model.specification()
        text = spec.sources[0].text
        assert r"\S^d_\mathrm{sets}&: S" in text
        assert not hasattr(ns, "SetCover")

    def test_load_notebook_model_classes(self):
        ns = load_notebook_models(
            "notebooks/set-cover.ipynb",
            root=__file__,
            include_classes=True,
        )
        assert issubclass(ns.SetCover, Model)

    def test_load_notebook_model_symbols(self):
        ns = load_notebook_models(
            "notebooks/set-cover.ipynb",
            root=__file__,
            include_symbols=["SetCover"],
        )
        assert issubclass(ns.SetCover, Model)
