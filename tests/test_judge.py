"""LLM judge test: natural-language criteria for the rules answer (Q3)."""

import pytest

from tests.judge import assert_criteria
from tests.utils import collect_tools, run_agent_test

criteria = [
    "the first tool call in the trace is search_rules, before any other tool",
    "the answer cites '(Player's Handbook, D&D 5E)' as the source of the rule",
    "the answer states that an opportunity attack costs the character's reaction "
    "and is triggered when a creature moves out of the attacker's reach",
]


@pytest.mark.asyncio
async def test_opportunity_attack_answer(agent):
    result = await run_agent_test(
        agent,
        "How does an opportunity attack work? What triggers it and what does it cost?",
    )

    # Hard check on the trace the judge is looking at (cheap, no LLM).
    tool_calls = collect_tools(result.all_messages())
    assert tool_calls and tool_calls[0].name == "search_rules"

    await assert_criteria(result, criteria)
