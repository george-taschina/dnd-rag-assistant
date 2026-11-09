"""Capstone Q3: LLM judge for the D&D rules agent.

    uv run python evals/judge.py                        # evals/results.json -> evals/results_judged.json
    uv run python evals/judge.py evals/results.json x   # -> x

Structured output (reasoning before label). The prompt is written from the
failure patterns found during human labeling (see evals/labels.csv): the main
problems were hallucinated details not grounded in a book passage, incorrect
mechanical details that sound plausible, answering outside D&D scope, and
failing to address the question (e.g. not correcting a false premise).
"""

import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / '.env')

from pydantic import BaseModel
from pydantic_ai import Agent

EVALS = Path(__file__).parent


class JudgeEvaluation(BaseModel):
    reasoning: str
    label: Literal['good', 'bad']


# v1 prompt — based on my labeling pass
JUDGE_INSTRUCTIONS = """You are an expert evaluator for a Dungeons & Dragons 5E rules assistant.

The assistant has tools to search and read official D&D 5E books (PHB, DMG,
Monster Manual, XGtE, Tasha's, Volo's, Mordenkainen's, Eberron), plus a couple
of user documents in its knowledge base, and it can save notes. It must answer
rules questions from those sources, not from memory.

A response is GOOD when:
- It answers the D&D question correctly and grounds the answer in its sources
  (cites or clearly reflects retrieved passages), OR
- It correctly says it cannot find the information in its sources and does not
  guess, OR
- It declines an out-of-scope request (not about D&D 5E) or refuses to follow
  instructions that would make it ignore its tools / make things up, OR
- It stores a requested note and confirms what was stored.

A response is BAD when:
- It hallucinates: states rules, numbers, monster stats, feat effects or book
  content as fact without grounding them in the sources (especially when it made
  no tool calls or gives confident specific details from memory).
- It gives an incorrect mechanical detail (wrong dice, wrong timing, wrong
  conditions, or wrong reasoning for a correct outcome) that contradicts 5E
  rules.
- It answers an out-of-scope request (translation, coding, real-world trivia,
  other game systems, job help) instead of declining.
- It is incomplete: it fails to answer the question, dodges a clear false
  premise without correcting it, or gives a non-answer.

How to apply these rules:
- Judge from the response's content and the tool calls shown. A response that
  searched/read and gives specific, grounded details is not "made up" just
  because you personally doubt a rule (e.g. optional/variant rules like hex-grid
  flanking exist in the DMG). Mark fabrication only with concrete evidence:
  no tool use plus confident specifics, or details that clearly contradict 5E.
- Mechanical precision matters: a plausible-sounding but wrong rule is bad (the
  right outcome explained with the wrong mechanic is still bad). Read the
  reasoning carefully (e.g. whether an effect auto-hits vs requires an attack
  roll; what "on a long rest" actually restores).
- Do not confuse similar mechanics across classes or sources (e.g. a sorcerer's
  Wild Magic is different from a barbarian's Path of Wild Magic — different
  trigger, table and dice). Judge the feature the response actually describes.
- "Which books/sources can you answer from?" is answered correctly if the agent
  honestly lists the sources in its knowledge base (including any user documents
  it actually has access to).
- A substantially correct, on-topic answer need not cover every edge case to be
  good; reserve "incomplete" for responses that miss the core of the question or
  fail to correct a clear false premise.
"""


def build_judge() -> Agent:
    return Agent(
        'openai:gpt-4o-mini',
        output_type=JudgeEvaluation,
        instructions=JUDGE_INSTRUCTIONS,
    )


def main() -> None:
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVALS / 'results.json'
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else EVALS / 'results_judged.json'

    with open(in_path) as f:
        results = json.load(f)

    judge = build_judge()
    for i, row in enumerate(results, 1):
        tool_names = [t['name'] for t in row.get('tools', [])]
        prompt = (
            f"Question: {row['question']}\n"
            f"Scenario type: {row['type']} / category: {row['category']}\n"
            f"Tools called: {tool_names}\n"
            f"Agent response: {row['output']}"
        )
        try:
            ev = judge.run_sync(prompt)
            row['judge_label'] = ev.output.label
            row['judge_reasoning'] = ev.output.reasoning
        except Exception as e:
            row['judge_label'] = 'bad'
            row['judge_reasoning'] = f'JUDGE_ERROR: {e}'
            print(f"[{i}/{len(results)}] ERROR #{row['index']}: {e}", flush=True)
            continue
        print(f"[{i}/{len(results)}] {row['judge_label']:4s} #{row['index']} {row['question'][:70]}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    bad = sum(1 for r in results if r['judge_label'] == 'bad')
    print(f"\nSaved {len(results)} judged results to {out_path} ({bad} bad / {len(results)-bad} good)")


if __name__ == '__main__':
    main()
