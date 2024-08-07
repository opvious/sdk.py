import os
import pytest


from opvious.modeling import Model
from opvious.specifications import (
    LocalSpecification,
    load_notebook_models,
)


class TestSpecifications:
    def test_local_globs_file_root(self):
        spec = LocalSpecification.globs(
            "**/*bounded.md", "sources/sudo*", root=__file__
        )
        assert len(spec.sources) == 3

    def test_local_globs_file_nested_root(self):
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "sources"
        )
        spec = LocalSpecification.globs("*bounded.md", root=root)
        assert len(spec.sources) == 2

    @pytest.mark.asyncio
    async def test_inline_sources(self):
        spec = LocalSpecification.globs("sources/bounded.md", root=__file__)
        assert len(spec.sources) == 1
        assert "greaterThanBound" in spec.sources[0].text

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
