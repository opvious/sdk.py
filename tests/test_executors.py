import opvious
import os
import pytest


from opvious.executors.aiohttp import AiohttpExecutor
from opvious.executors.urllib import UrllibExecutor


AUTHORIZATION = os.environ.get("OPVIOUS_AUTHORIZATION")


API_URL = "https://api.beta.opvious.io"


@pytest.mark.skipif(not AUTHORIZATION, reason="No access token detected")
class TestExecutors:
    _authorization = (
        AUTHORIZATION if " " in AUTHORIZATION else f"Bearer {AUTHORIZATION}"
    )
    _executors = [
        AiohttpExecutor(api_url=API_URL, authorization=_authorization),
        UrllibExecutor(api_url=API_URL, authorization=_authorization),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_ok(self, executor):
        res = await executor.execute("@FetchMember", {})
        assert list(res.keys()) == ["me"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_missing_argument(self, executor):
        with pytest.raises(opvious.ApiError) as info:
            await executor.execute("@PaginateFormulations", {})
        assert info.value.status == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_not_found(self, executor):
        with pytest.raises(opvious.ApiError) as info:
            await executor.execute(
                "@CancelAttempt",
                {
                    "uuid": "00000000-0000-0000-0000-000000000000",
                },
            )
        assert info.value.status == 404
