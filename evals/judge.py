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

The assistant has tools (`search_rules`, `read_passage`, `list_sources`, `save_note`) to search and read a fixed knowledge base of 10 documents: 8 official D&D 5E rulebooks (Player's Handbook, Dungeon Master's Guide, Monster Manual, Xanathar's Guide to Everything, Tasha's Cauldron of Everything, Volo's Guide to Monsters, Mordenkainen's Tome of Foes, Eberron: Rising from the Last War) plus 2 user campaign/character documents. It must answer rules questions from those sources, not from memory.

A response is GOOD when:
1. It answers the D&D question accurately and grounds the answer in its sources (calling `search_rules` and/or `read_passage` and reflecting the rules accurately), OR
2. For "which books/sources can you answer from?", it calls `list_sources` and lists all sources in its knowledge base (including the 2 user documents—listing what is actually in the KB is required and correct, NEVER penalize it), OR
3. It declines an out-of-scope request (not about D&D 5E) or refuses to fabricate/guess, OR
4. It stores a requested note via `save_note` and confirms what was stored.

A response is BAD when ANY of these apply:
1. Hallucination / ungrounded specifics: It states specific mechanical rules, feat effects, or monster descriptions from memory without reading the relevant passages, OR makes claims with 0 search/read tool calls on a rules/monsters question.
2. Incorrect mechanical detail: It states a rule that contradicts D&D 5E mechanics, even if it cites a book or gets the high-level yes/no right. Watch out for common 5E rule traps:
   - Long rest Hit Dice recovery: A character regains spent Hit Dice up to HALF their total number of Hit Dice (minimum 1)—they do NOT regain "all spent Hit Dice plus extra". Saying you regain all spent Hit Dice on a long rest is BAD.
   - Shield vs Magic Missile: Magic Missile hits automatically (NO attack roll); Shield blocks it because Shield's text explicitly says you take no damage from Magic Missile. Claiming Magic Missile requires an attack roll is BAD.
   - Rogue Sneak Attack on reactions: Sneak Attack is "once per turn" (which includes other creatures' turns on a reaction like an opportunity attack), NOT "only on the rogue's turn". Saying reactions cannot trigger Sneak Attack is BAD.
3. Wrong scope: It answers an out-of-scope request (translation, coding, real-world trivia, non-5E games) instead of declining.
4. Incomplete: It fails to answer the question, gets confused by a truncated prompt, or dodges a false premise without correcting it.

Important calibration rules (avoid false positives):
- Trust retrieved book rules when tools (`search_rules`, `read_passage`) were called! Do NOT mark a grounded response BAD just because an optional rule or subclass feature sounds unfamiliar to you:
  - The Dungeon Master's Guide DOES contain optional "Flanking on Hexes" rules with exact hex counts by creature size (2/4/5/6 hexes). A response explaining that after calling `search_rules` + `read_passage` is GOOD.
  - "Wild Magic Surge" is ALSO the name of the 3rd-level feature of the Barbarian's Path of Wild Magic in *Tasha's Cauldron of Everything* (rolling a d8 on the Wild Magic table when entering a rage), not just the PHB Sorcerer feature. Describing the Tasha's Barbarian feature after searching is GOOD.
  - Drinking a standard potion takes an action, while noting that specific potions (like Potion of Fire Breath) have bonus-action activation effects in their description is accurate and GOOD.
  - Explaining wizard spell-scroll copying mechanics (must be on wizard list, time/cost per level) after reading passages is GOOD; do not nitpick secondary book citations if the mechanics are right.
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
