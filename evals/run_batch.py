"""Capstone Q1: run the D&D agent on all evaluation scenarios in batch.

    uv run python evals/run_batch.py

Reads evals/scenarios.csv, runs each question as an independent conversation
(no shared history), and saves question, category, type, response, tool calls,
tokens, cost and latency to evals/results.json.
"""

import csv
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # import the project modules (tools, doc_agent, ...)

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / '.env')

from cost_tracker import cost_usd
from doc_agent import DndAgentConfig, create_agent, format_tool_calls
from tools import create_dnd_tools

EVALS = Path(__file__).parent
SCENARIOS = EVALS / 'scenarios.csv'
RESULTS = EVALS / 'results.json'


def main() -> None:
    with open(SCENARIOS) as f:
        scenarios = list(csv.DictReader(f))

    agent = create_agent(DndAgentConfig(), create_dnd_tools())
    results = []

    for i, sc in enumerate(scenarios, 1):
        question = sc['question']
        print(f"[{i}/{len(scenarios)}] [{sc['type']}] {question}")

        started = time.time()
        result = agent.run_sync(question)
        elapsed = round(time.time() - started, 2)

        usage = result.usage
        tools = format_tool_calls(result)
        results.append({
            'index': i,
            'question': question,
            'category': sc['category'],
            'type': sc['type'],
            'output': result.output,
            'tools': tools,
            'tokens': {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'total_tokens': usage.total_tokens,
            },
            'cost': cost_usd('openai-responses:gpt-4o-mini',
                             usage.input_tokens, usage.output_tokens),
            'execution_time': elapsed,
        })
        print(f"    {len(tools)} tool calls, {usage.input_tokens} in / "
              f"{usage.output_tokens} out, {elapsed}s")

    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    total_cost = sum(r['cost'] for r in results)
    total_time = sum(r['execution_time'] for r in results)
    print(f"\nSaved {len(results)} results to {RESULTS}")
    print(f"Total: ${total_cost:.4f}, {total_time:.0f}s")


if __name__ == '__main__':
    main()
