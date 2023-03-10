import opvious
import os
import pytest


from opvious.executors.aiohttp import AiohttpExecutor
from opvious.executors.common import execute_graphql_query
from opvious.executors.urllib import UrllibExecutor


AUTHORIZATION = os.environ.get("OPVIOUS_AUTHORIZATION")


DOMAIN = os.environ.get("OPVIOUS_DOMAIN", "alpha.opvious.io")


@pytest.mark.skipif(not AUTHORIZATION, reason="No access token detected")
class TestExecutors:
    _authorization = (
        AUTHORIZATION if " " in AUTHORIZATION else f"Bearer {AUTHORIZATION}"
    )
    _executors = [
        AiohttpExecutor(
            api_url=f"https://api.{DOMAIN}", authorization=_authorization
        ),
        UrllibExecutor(
            api_url=f"https://api.{DOMAIN}", authorization=_authorization
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_ok(self, executor):
        res = await execute_graphql_query(executor, "@FetchMember")
        assert list(res.keys()) == ["me"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_missing_argument(self, executor):
        with pytest.raises(opvious.ApiError) as info:
            await execute_graphql_query(executor, "@PaginateFormulations")
        assert info.value.status == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_not_found(self, executor):
        with pytest.raises(opvious.ApiError) as info:
            await execute_graphql_query(
                executor,
                "@CancelAttempt",
                {"uuid": "00000000-0000-0000-0000-000000000000"},
            )
        assert info.value.status == 404
