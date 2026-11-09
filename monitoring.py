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
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import logfire
from dotenv import load_dotenv

load_dotenv()

FEEDBACK_FILE = Path(__file__).resolve().parent / "data" / "feedback.json"

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


def record_feedback(
    session_id: str,
    question: str,
    answer: str,
    rating: int,
    comment: str = "",
    lf_ctx=None,
) -> dict:
    """Record thumbs-up (+1) / thumbs-down (-1) user feedback in Logfire and
    append it to data/feedback.json so it is persisted both in the observability
    backend and locally.
    """
    label = "positive" if rating > 0 else "negative"
    if lf_ctx is not None:
        with logfire.attach_context(lf_ctx):
            logfire.info(
                "user_feedback",
                session_id=session_id,
                question=question,
                rating=rating,
                label=label,
                comment=comment,
            )
    else:
        logfire.info(
            "user_feedback",
            session_id=session_id,
            question=question,
            rating=rating,
            label=label,
            comment=comment,
        )

    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    entries = json.loads(FEEDBACK_FILE.read_text()) if FEEDBACK_FILE.exists() else []
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question,
        "answer_preview": answer[:240],
        "rating": rating,
        "label": label,
        "comment": comment,
    }
    entries.append(entry)
    FEEDBACK_FILE.write_text(json.dumps(entries, indent=2))
    return entry


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
