"""``intercept`` — mark a callable as recordable / replayable.

Outside an active session the wrapper is a transparent pass-through, so you can
leave ``intercept`` calls in production code with no behavioural change.
Handles plain functions, ``async def`` coroutines, and async generators.
"""

from __future__ import annotations

import functools
import inspect


def intercept(fn=None, *, name=None, kind="call", stream=None, key=None):
    """Wrap a sync / async / async-generator callable for record & replay.

    Parameters
    ----------
    name:  label stored on each step (defaults to the callable's ``__name__``).
    kind:  free-form category, e.g. ``"llm"`` or ``"tool"``.
    stream: force streaming handling instead of auto-detecting.
    key:   optional stable discriminator (value or callable of the call args).
           Needed only when the *same* call ran concurrently and legitimately
           returned *different* results.
    """

    def wrap(func):
        call_name = name or getattr(func, "__name__", "call")

        def _resolve_key(args, kwargs):
            return key(*args, **kwargs) if callable(key) else key

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def agwrapper(*args, **kwargs):
                from .session import current_session

                sess = current_session()
                if sess is None:
                    async for chunk in func(*args, **kwargs):
                        yield chunk
                    return
                stream_iter = await sess.on_call_agen(
                    func, call_name, kind, args, kwargs, stream, _resolve_key(args, kwargs)
                )
                async for chunk in stream_iter:
                    yield chunk

            agwrapper.__wrapped__ = func
            return agwrapper

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args, **kwargs):
                from .session import current_session

                sess = current_session()
                if sess is None:
                    return await func(*args, **kwargs)
                return await sess.on_call_async(
                    func, call_name, kind, args, kwargs, stream, _resolve_key(args, kwargs)
                )

            awrapper.__wrapped__ = func
            return awrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from .session import current_session

            sess = current_session()
            if sess is None:
                return func(*args, **kwargs)
            return sess.on_call_sync(
                func, call_name, kind, args, kwargs, stream, _resolve_key(args, kwargs)
            )

        wrapper.__wrapped__ = func
        return wrapper

    return wrap if fn is None else wrap(fn)
