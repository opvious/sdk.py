import opvious
import opvious.executors
import pytest


TOKEN = opvious.ClientSetting.TOKEN.read()


ENDPOINT = opvious.ClientSetting.ENDPOINT.read()


@pytest.mark.skipif(not TOKEN, reason="No access token detected")
class TestExecutors:
    _authorization = opvious.executors.authorization_header(TOKEN)
    _executors = [
        opvious.executors.aiohttp_executor(
            endpoint=ENDPOINT, authorization=_authorization
        ),
        opvious.executors.aiohttp_executor(
            endpoint=ENDPOINT, authorization=_authorization
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
        assert info.value.status == 200
        assert "ERR_INVALID" in info.value.reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize("executor", _executors)
    async def test_execute_not_found(self, executor):
        with pytest.raises(opvious.executors.ExecutorError) as info:
            await executor.execute_graphql_query(
                "@CancelAttempt",
                {"uuid": "00000000-0000-0000-0000-000000000000"},
            )
        assert info.value.status == 200
        assert "ERR_UNKNOWN_ATTEMPT" in info.value.reason

    @pytest.mark.asyncio
    async def test_fetch_text(self):
        text = await opvious.executors.fetch_text(
            "https://gist.githubusercontent.com/mtth/82a21baacba4827bc1710b7526775315/raw/4e2b17c1509dfd40595f47167e8d859fa4ec8334/sudoku.md"  # noqa
        )
        assert "defining" in text

    @pytest.mark.asyncio
    async def test_fetch_csv(self):
        df = await opvious.executors.fetch_csv(
            "https://gist.githubusercontent.com/mtth/f59bdde8694223c06f77089b71b48d17/raw/6f1568cb2ff69450f06e3b8045d504af74bb701f/fpl-2023-07-26.csv"  # noqa
        )
        assert len(df) == 605
