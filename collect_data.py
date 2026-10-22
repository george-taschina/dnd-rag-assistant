"""Q3: collect real interaction data by driving the Streamlit app.

    uv run python collect_data.py

Uses streamlit.testing.v1.AppTest to send queries through app.py itself
(one AppTest object = one browser session = one Logfire trace tree), then
summarises what landed in Logfire. Results are saved to
data/usage_collection.json.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from streamlit.testing.v1 import AppTest

from logfire.query_client import LogfireQueryClient

SESSIONS = {
    "typical-rules": [
        "How does an opportunity attack work? What triggers it and what does it cost?",
        "What are the prerequisites for becoming a paladin?",
        "Can a rogue use Sneak Attack more than once per turn?",
    ],
    "edge-cases": [
        "What is the airspeed velocity of an unladen swallow?",
        "Remember this house rule: critical hits deal maximum weapon damage plus one die.",
        "Come si calcola il bonus di competenza? (asked in Italian)",
    ],
    "follow-ups": [
        "Does the Shield spell work against Magic Missile?",
        "What about against a normal arrow?",
        "Which books are you able to answer from?",
    ],
    "character-build": [
        "How does grappling work in 5E?",
        "I want a level 5 divination wizard - which abilities and spells do I get?",
        "Does expertise stack with Advantage on the same check?",
    ],
}

OUT = Path("data/usage_collection.json")


def collect() -> list[dict]:
    results = []
    for session_name, queries in SESSIONS.items():
        at = AppTest.from_file("app.py", default_timeout=600)
        at.run()  # first run: builds session state + the dnd_session span
        for q in queries:
            t0 = time.time()
            at.chat_input[0].set_value(q)
            at.run()
            elapsed = round(time.time() - t0, 1)

            messages = at.session_state["messages"]
            assistant = [m for m in messages if m.get("role") == "assistant"]
            last = assistant[-1] if assistant else {}
            results.append({
                "session": session_name,
                "question": q,
                "answer_chars": len(last.get("content", "")),
                "tools": [t["name"] for t in last.get("tools", [])],
                "seconds": elapsed,
            })
            print(f"  [{session_name}] {q[:60]!r} -> {len(last.get('tools', []))} tool calls, {elapsed}s")
    return results


def summarise_from_logfire(results: list[dict], since: datetime) -> dict:
    """Read back what actually landed in Logfire (dashboard side)."""
    client = LogfireQueryClient(read_token=os.environ["LOGFIRE_READ_TOKEN"])

    sessions = client.query_json_rows(
        sql="""
        SELECT trace_id, attributes
        FROM records
        WHERE span_name = 'dnd_session'
        ORDER BY start_timestamp
        """,
        min_timestamp=since,
    )["rows"]

    per_session = []
    for s in sessions:
        tid = s["trace_id"]
        spans = client.query_json_rows(
            sql=f"""
            SELECT span_name, attributes
            FROM records
            WHERE trace_id = '{tid}'
            ORDER BY start_timestamp
            """,
            min_timestamp=since,
        )["rows"]
        tools, input_tokens, output_tokens, runs = [], 0, 0, 0
        for sp in spans:
            attrs = sp["attributes"] if isinstance(sp["attributes"], dict) else json.loads(sp["attributes"])
            if sp["span_name"].startswith("execute_tool "):
                tools.append(sp["span_name"].split(" ", 1)[1])
            elif sp["span_name"].startswith("invoke_agent "):
                runs += 1
                input_tokens += attrs.get("gen_ai.aggregated_usage.input_tokens", 0) or 0
                output_tokens += attrs.get("gen_ai.aggregated_usage.output_tokens", 0) or 0
        cost = (input_tokens / 1e6) * 0.15 + (output_tokens / 1e6) * 0.60
        per_session.append({
            "trace_id": tid, "runs": runs, "tools": tools,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })
    return {"sessions_in_logfire": per_session}


if __name__ == "__main__":
    started = datetime.now(timezone.utc) - timedelta(seconds=5)
    print("collecting interactions through the app…")
    results = collect()
    print(f"\ncollected {len(results)} interactions; reading traces back from Logfire…")
    time.sleep(15)  # let the exporter flush
    payload = {"interactions": results, **summarise_from_logfire(results, started)}
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"saved to {OUT}")

    sessions = payload["sessions_in_logfire"]
    print(f"\n{len(sessions)} session traces in Logfire")
    for s in sessions:
        print(f"  {s['trace_id'][:16]}… runs={s['runs']} in={s['input_tokens']} out={s['output_tokens']} "
              f"${s['cost_usd']:.5f} tools={s['tools']}")
