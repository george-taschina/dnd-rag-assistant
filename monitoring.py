"""Logfire monitoring setup for the D&D agent (monitoring capstone).

Imported by the app and the data-collection scripts - not by doc_agent.py -
so the agent module and its tests stay instrumentation-free.

Patterns follow the course references (documentation-agent/app.py,
cost_tracker.py) plus the "joining multiple runs" and "tracking user
feedback" lessons.
"""

import asyncio
import os
import threading
from contextlib import contextmanager

import logfire
from dotenv import load_dotenv

load_dotenv()

# Configure once per process. With no token the SDK stays silent, so tests
# and plain CLI runs are unaffected.
logfire.configure(
    token=os.getenv("LOGFIRE_WRITE_TOKEN"),
    send_to_logfire="if-token-present",
    console=False,
)
logfire.instrument_pydantic_ai()


@contextmanager
def dnd_session(session_id: str):
    """One session = one Logfire trace tree; the span stays open for the
    duration of the block (scripts), so everything inside nests under it.

    Returns the logfire context - Streamlit apps can save it and re-attach
    later calls with logfire.attach_context() (the "joining multiple runs
    under one span tree" lesson), since a with-block cannot stay open
    across Streamlit reruns.
    """
    with logfire.span("dnd_session", session_id=session_id):
        yield logfire.get_context()


_LOOP: asyncio.AbstractEventLoop | None = None


def get_loop() -> asyncio.AbstractEventLoop:
    """Process-wide event loop for agent runs (created on first use)."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
        threading.Thread(target=_LOOP.run_forever, name="agent-loop", daemon=True).start()
    return _LOOP


def run_async(coro, timeout: float = 300):
    """Run an agent coroutine on the shared loop (thread-safe).

    The agent's HTTP client pool must always be used from the SAME event
    loop: a fresh asyncio.run() per interaction hands the pool a second
    (already closed) loop and dies with "Event loop is closed" on the
    next question. Call inside logfire.attach_context(...) - the calling
    thread's contextvars are captured and run with the coroutine.
    """
    return asyncio.run_coroutine_threadsafe(coro, get_loop()).result(timeout)
