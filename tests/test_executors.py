import opvious
import opvious.executors
import pytest


TOKEN = opvious.ClientSetting.TOKEN.read()


DOMAIN = opvious.ClientSetting.DOMAIN.read()


@pytest.mark.skipif(not TOKEN, reason="No access token detected")
class TestExecutors:
    _authorization = TOKEN if " " in TOKEN else f"Bearer {TOKEN}"
    _executors = [
        opvious.executors.aiohttp_executor(
            root_url=f"https://api.{DOMAIN}", authorization=_authorization
        ),
        opvious.executors.aiohttp_executor(
            root_url=f"https://api.{DOMAIN}", authorization=_authorization
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_ok(self, executor):
        res = await executor.execute_graphql_query("@FetchMember")
        assert list(res.keys()) == ["me"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_missing_argument(self, executor):
        with pytest.raises(opvious.executors.ExecutorError) as info:
            await executor.execute_graphql_query("@PaginateFormulations")
        assert info.value.status == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_not_found(self, executor):
        with pytest.raises(opvious.executors.ExecutorError) as info:
            await executor.execute_graphql_query(
                "@CancelAttempt",
                {"uuid": "00000000-0000-0000-0000-000000000000"},
            )
        assert info.value.status == 404
