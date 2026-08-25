"""Job-kind registry.

A handler is anything with a ``kind`` and a ``run(spec, ctx, tier)``. Adding a
capability means writing one class and listing it here.
"""

from __future__ import annotations

from .ask import AskHandler
from .base import Ctx, Handler, HandlerResult
from .digest import DigestHandler
from .grep import GrepHandler
from .review import ReviewHandler
from .transcribe import TranscribeHandler
from .transform import TransformHandler

_HANDLERS: dict[str, Handler] = {}


def register(handler: Handler) -> Handler:
    _HANDLERS[handler.kind] = handler
    return handler


for _h in (
    DigestHandler(),
    GrepHandler(),
    AskHandler(),
    TransformHandler(),
    ReviewHandler(),
    TranscribeHandler(),
):
    register(_h)


def get_handler(kind: str) -> Handler:
    try:
        return _HANDLERS[kind]
    except KeyError:
        known = ", ".join(sorted(_HANDLERS))
        raise KeyError(f"unknown job kind {kind!r}; available: {known}") from None


def kinds() -> list[str]:
    return sorted(_HANDLERS)


__all__ = ["Ctx", "Handler", "HandlerResult", "get_handler", "kinds", "register"]
