"""Q1 evidence: run the agent on one query, then read the trace back.

    uv run python check_trace.py

Mirrors the course's documentation-agent/otel_test.py: the run is sent to
Logfire, then fetched back with the query client (read token) and
reconstructed into an AgentRunResult with trace_replay.
"""

import os
import time

import asyncio
from datetime import datetime, timezone

import monitoring  # noqa: F401  (configures + instruments Logfire)
from dotenv import load_dotenv

load_dotenv()

from logfire.query_client import LogfireQueryClient

from doc_agent import DndAgentConfig, create_agent, format_tool_calls
from tools import create_dnd_tools
from trace_replay import fetch_trace, trace_to_run_result

QUESTION = "How does an opportunity attack work? What triggers it and what does it cost?"


async def main() -> None:
    tools = create_dnd_tools()
    agent = create_agent(DndAgentConfig(), tools)

    started = time.time()
    with monitoring.dnd_session("check-trace"):
        result = await agent.run(QUESTION)

    print(f"QUESTION: {QUESTION}")
    print(f"TOOLS: {[t['name'] for t in format_tool_calls(result)]}")
    print(f"ANSWER (first 200 chars): {result.output[:200]}")
    print(f"USAGE: {result.usage}")

    # let the exporter flush, then read the trace back from Logfire
    time.sleep(10)

    client = LogfireQueryClient(read_token=os.environ["LOGFIRE_READ_TOKEN"])
    rows = client.query_json_rows(
        sql="""
        SELECT trace_id, span_name, start_timestamp
        FROM records
        WHERE span_name = 'dnd_session'
           OR span_name LIKE 'invoke_agent %'
           OR span_name LIKE 'execute_tool %'
           OR span_name LIKE 'chat %'
        ORDER BY start_timestamp DESC
        LIMIT 12
        """,
        min_timestamp=datetime.fromtimestamp(started - 5, tz=timezone.utc),
    )["rows"]

    print("\n--- recent spans in Logfire ---")
    for r in rows:
        print(f"  {r['trace_id'][:16]}…  {r['span_name']}")

    trace_id = rows[0]["trace_id"] if rows else None
    if trace_id:
        trace = fetch_trace(trace_id, client)
        run = trace_to_run_result(trace)
        print(f"\n--- replayed trace {trace_id} ---")
        print(f"  messages: {len(run.all_messages())}")
        print(f"  tokens:   {trace.input_tokens} in / {trace.output_tokens} out")
        print(f"  output starts: {run.output[:120]!r}")
        print(f"\nOpen in the UI: https://logfire-eu.pydantic.dev/george-taschina/starter-project/traces?trace_id={trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
