import pytest
from typing import Any, cast


from opvious.specifications import LocalSpecification


class TestSpecifications:
    def test_local_globs(self):
        spec = LocalSpecification.globs(
            "**/*bounded.md", "sources/sudo*", root_dir="tests"
        )
        assert len(spec.paths) == 3

    @pytest.mark.asyncio
    async def test_local_sources(self):
        spec = LocalSpecification(["tests/sources/bounded.md"])
        sources = await spec.fetch_sources(cast(Any, None))
        assert len(sources) == 1
        assert "greaterThanBound" in sources[0]
