from __future__ import annotations

import contextvars
from typing import Any, Protocol


_active_renderer: Any = contextvars.ContextVar("renderer")


class Renderer:
    def __init__(self) -> None:
        self.__token: Any = None

    def __enter__(self) -> None:
        if _active_renderer.get(None):
            raise Exception("Renderer already active")
        self.__token = _active_renderer.set(self)

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        _active_renderer.reset(self.__token)
        self.__token = None

    def render_identifier(self, identifier: Identifier) -> str:
        raise NotImplementedError()


class Identifier:
    def render(self) -> str:
        renderer = _active_renderer.get()
        return renderer.render_identifier(self)


class HasIdentifier(Protocol):
    @property
    def identifier(self) -> Identifier:
        raise NotImplementedError()
