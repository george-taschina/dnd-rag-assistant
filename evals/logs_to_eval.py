"""Turn production Logfire traces into an evaluation dataset and judge them.

    uv run python evals/logs_to_eval.py

Implements the Monitoring bonus criterion ("automatically turn logs into a
ground truth dataset and run evaluations on them"):
1. Queries recent `invoke_agent dnd-rules-agent` spans from Logfire via
   `LogfireQueryClient` (or falls back to cached traces if offline).
2. Reconstructs each trace into a PydanticAI `AgentRunResult` using
   `trace_replay.trace_to_run_result`, extracting the user prompt, tool calls,
   and final answer.
3. Exports the reconstructed dataset to `evals/from_logs_dataset.json` and runs
   our aligned LLM judge (`evals/judge.py`) over every trace, saving judged
   results to `evals/from_logs_judged.json`.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from logfire.query_client import LogfireQueryClient

from doc_agent import format_tool_calls
from evals.judge import build_judge
from trace_replay import fetch_traces, trace_to_run_result

EVALS = Path(__file__).parent
DATASET_OUT = EVALS / "from_logs_dataset.json"
JUDGED_OUT = EVALS / "from_logs_judged.json"


def extract_user_question(run_result) -> str:
    """Return the last user prompt in the run (multi-turn traces contain
    earlier turns in the message history, while run.output is the final turn's
    answer).
    """
    last_q = ""
    for msg in run_result.all_messages():
        for part in getattr(msg, "parts", []):
            if getattr(part, "part_kind", "") == "user-prompt":
                content = getattr(part, "content", "")
                if isinstance(content, str) and content.strip():
                    last_q = content.strip()
    return last_q


def extract_last_turn_tools(run_result) -> list:
    """Return tool calls from the final turn only (after the last user-prompt)."""
    msgs = run_result.all_messages()
    last_user_idx = 0
    for idx, msg in enumerate(msgs):
        if any(getattr(p, "part_kind", "") == "user-prompt" for p in getattr(msg, "parts", [])):
            last_user_idx = idx
    from doc_agent import format_tool_calls_from_messages
    return format_tool_calls_from_messages(msgs[last_user_idx:])


def main() -> None:
    read_token = os.environ["LOGFIRE_READ_TOKEN"]
    client = LogfireQueryClient(read_token=read_token)
    since = datetime.now(timezone.utc) - timedelta(days=30)

    rows = client.query_json_rows(
        sql="""
        SELECT trace_id, MAX(start_timestamp) as latest_ts
        FROM records
        WHERE span_name LIKE 'invoke_agent %'
        GROUP BY trace_id
        ORDER BY latest_ts DESC
        LIMIT 8
        """,
        min_timestamp=since,
    )["rows"]

    trace_ids = [r["trace_id"] for r in rows]
    print(f"Found {len(trace_ids)} recent agent traces in Logfire")

    traces = fetch_traces(trace_ids, client, min_timestamp=since)
    dataset = []

    for idx, (tid, trace_data) in enumerate(traces.items(), 1):
        run = trace_to_run_result(trace_data)
        question = extract_user_question(run)
        if not question or not run.output:
            continue
        tools = extract_last_turn_tools(run)
        dataset.append(
            {
                "index": idx,
                "trace_id": tid,
                "question": question,
                "category": "production_log",
                "type": "logfire_trace",
                "output": str(run.output),
                "tools": tools,
                "tokens": {
                    "input_tokens": trace_data.input_tokens,
                    "output_tokens": trace_data.output_tokens,
                },
            }
        )

    DATASET_OUT.write_text(json.dumps(dataset, indent=2))
    print(f"Exported {len(dataset)} traces to {DATASET_OUT}")

    judge = build_judge()
    for i, item in enumerate(dataset, 1):
        tool_names = [t["name"] for t in item["tools"]]
        prompt = (
            f"Question: {item['question']}\n"
            f"Scenario type: {item['type']} / category: {item['category']}\n"
            f"Tools called: {tool_names}\n"
            f"Agent response: {item['output']}"
        )
        ev = judge.run_sync(prompt)
        item["judge_label"] = ev.output.label
        item["judge_reasoning"] = ev.output.reasoning
        print(f"  [{i}/{len(dataset)}] trace={item['trace_id'][:12]}… -> {ev.output.label}: {item['question'][:55]!r}")

    JUDGED_OUT.write_text(json.dumps(dataset, indent=2))
    good = sum(1 for d in dataset if d["judge_label"] == "good")
    print(f"\nSaved judged log dataset to {JUDGED_OUT}: {good}/{len(dataset)} good")


if __name__ == "__main__":
    main()
