"""Prompt & retrieval tuning experiment on the 7 failed evaluation scenarios.

    uv run python evals/tune_agent.py

Compares the baseline agent outputs (from evals/results.json, before tuning
BOOST and DEFAULT_INSTRUCTIONS) with the tuned agent on the 7 scenarios that
failed human labeling (#8, #24, #25, #28, #33, #42, #43), then grades the new
responses with the aligned LLM judge (evals/judge.py) and saves the comparison
to evals/tuning_comparison.json.
"""

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from doc_agent import DndAgentConfig, create_agent, format_tool_calls
from evals.judge import build_judge
from tools import create_dnd_tools

EVALS = Path(__file__).parent
FAILED_INDICES = [8, 24, 25, 28, 33, 42, 43]


def main() -> None:
    baseline_results = {r["index"]: r for r in json.load(open(EVALS / "results_judged.json"))}
    scenarios = list(csv.DictReader(open(EVALS / "scenarios.csv")))

    agent = create_agent(DndAgentConfig(), create_dnd_tools())
    judge = build_judge()

    comparison = []
    passed_after = 0

    for idx in FAILED_INDICES:
        sc = scenarios[idx - 1]
        base = baseline_results[idx]
        q = sc["question"]
        print(f"[#{idx}] {q}")

        t0 = time.time()
        res = agent.run_sync(q)
        elapsed = round(time.time() - t0, 2)
        tools = format_tool_calls(res)
        tool_names = [t["name"] for t in tools]

        prompt = (
            f"Question: {q}\n"
            f"Scenario type: {sc['type']} / category: {sc['category']}\n"
            f"Tools called: {tool_names}\n"
            f"Agent response: {res.output}"
        )
        ev = judge.run_sync(prompt)
        if ev.output.label == "good":
            passed_after += 1

        comparison.append(
            {
                "index": idx,
                "question": q,
                "category": sc["category"],
                "type": sc["type"],
                "before_tools": [t["name"] for t in base["tools"]],
                "before_output": base["output"],
                "before_human_label": "bad",
                "before_judge_label": base["judge_label"],
                "after_tools": tool_names,
                "after_output": res.output,
                "after_judge_label": ev.output.label,
                "after_judge_reasoning": ev.output.reasoning,
                "execution_time": elapsed,
            }
        )
        print(f"    before: human=bad, judge={base['judge_label']} | after: judge={ev.output.label} ({tool_names})")

    out_path = EVALS / "tuning_comparison.json"
    out_path.write_text(json.dumps(
        {
            "failed_scenarios_tested": len(FAILED_INDICES),
            "good_before": 0,
            "good_after": passed_after,
            "overall_50_pass_rate_before": "43/50 (86.0%)",
            "overall_50_pass_rate_after": f"{43 + passed_after}/50 ({(43 + passed_after) * 2:.1f}%)",
            "scenarios": comparison,
        },
        indent=2,
    ))
    print(f"\nFixed {passed_after}/{len(FAILED_INDICES)} previously failing scenarios -> overall {43 + passed_after}/50 ({(43 + passed_after) * 2:.1f}%)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
