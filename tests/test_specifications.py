import os
import pytest


from opvious.specifications import LocalSpecification, load_notebook_specification


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
        spec = LocalSpecification.globs("tests/sources/bounded.md")
        assert len(spec.sources) == 1
        assert "greaterThanBound" in spec.sources[0].text

    def test_load_notebook_specification(self):
        spec = load_notebook_specification(
            "notebooks/set-cover.ipynb",
            root=__file__
        )
        print(spec)
