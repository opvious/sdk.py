import os
import pytest
from typing import Any, cast


from opvious.specifications import LocalSpecification


class TestSpecifications:
    def test_local_globs_file_root(self):
        spec = LocalSpecification.globs(
            "**/*bounded.md", "sources/sudo*", root=__file__
        )
        assert len(spec.paths) == 3

    def test_local_globs_file_nested_root(self):
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "sources"
        )
        spec = LocalSpecification.globs("*bounded.md", root=root)
        assert len(spec.paths) == 2

    @pytest.mark.asyncio
    async def test_local_sources(self):
        spec = LocalSpecification(["tests/sources/bounded.md"])
        sources = await spec.fetch_sources(cast(Any, None))
        assert len(sources) == 1
        assert "greaterThanBound" in sources[0]
