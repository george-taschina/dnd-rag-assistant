"""Tests for the D&D rules agent: happy paths, tool usage, scope handling.

Each test runs the agent for real against the indexed rulebooks.
"""

import pytest

from tests.utils import collect_tools, run_agent_test


@pytest.mark.asyncio
async def test_rules_question_searches_first_and_cites_the_book(agent):
    """Scenario 1: happy path + tool order + citation."""
    result = await run_agent_test(
        agent, "How does an opportunity attack work? What triggers it and what does it cost?"
    )

    tool_calls = collect_tools(result.all_messages())
    assert tool_calls, "the agent answered without calling any tool"
    assert tool_calls[0].name == "search_rules", \
        f"expected search_rules first, got {[c.name for c in tool_calls]}"
    assert not any(c.name == "save_note" for c in tool_calls)

    answer = result.output
    assert "Player's Handbook" in answer, "answer is not cited to the Player's Handbook"
    assert "reaction" in answer.lower(), "the cost of the attack is missing"


@pytest.mark.asyncio
async def test_out_of_scope_question_is_refused(agent):
    """Scenario 2 (edge case): another game's rules must not be answered."""
    result = await run_agent_test(
        agent, "What are the best feats for a Pathfinder 2e alchemist?"
    )

    tool_calls = collect_tools(result.all_messages())
    # At most one search to confirm scope - it must not start browsing books.
    assert len(tool_calls) <= 1, [c.name for c in tool_calls]
    assert not any(c.name in ("read_passage", "save_note") for c in tool_calls)

    answer = result.output.lower()
    assert "pathfinder" in answer, "the agent did not acknowledge the question"
    assert "dungeons" in answer or "d&d" in answer, "the agent did not state its scope"


@pytest.mark.asyncio
async def test_list_sources_reports_the_indexed_books(agent):
    """Scenario 3: a scope question goes to list_sources, not to search_rules."""
    result = await run_agent_test(agent, "Which books do you actually have indexed?")

    tool_calls = collect_tools(result.all_messages())
    assert [c.name for c in tool_calls] == ["list_sources"], \
        f"expected exactly one list_sources call, got {[c.name for c in tool_calls]}"

    answer = result.output
    for title in ("Player's Handbook", "Monster Manual", "Eberron"):
        assert title in answer, f"{title} missing from the answer"
