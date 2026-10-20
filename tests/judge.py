"""LLM judge - the course's tests/judge.py, adapted to the D&D rules agent.

`evaluate_agent_performance` grades one agent run against natural-language
criteria and returns the structured feedback; `assert_criteria` additionally
asserts that every criterion passed.
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from tests.utils import collect_tools

judge_instructions = """
You are an expert judge evaluating the performance of an
AI agent.
""".strip()


class JudgeCriterion(BaseModel):
    """
    Evaluation of a single test requirement or behavioral rule.
    """
    criterion_description: str = Field(
        description="The specific requirement or rule that the agent is being evaluated against."
    )
    passed: bool = Field(
        description="Indicates whether the agent's response and actions successfully satisfied this requirement."
    )
    judgement: str = Field(
        description="A clear explanation of why the agent passed or failed, referencing specific evidence from the agent's output or tool calls."
    )


class JudgeFeedback(BaseModel):
    """
    The complete evaluation report from the judge agent, summarizing performance across all criteria.
    """
    criteria: list[JudgeCriterion] = Field(
        description="A collection of individual evaluations for each performance requirement provided in the test."
    )
    feedback: str = Field(
        description="A holistic summary of the agent's performance, providing overall context and identifying key strengths or failures."
    )


def create_judge_agent() -> Agent:
    return Agent(
        name="judge",
        model="openai:gpt-4o-mini",
        instructions=judge_instructions,
        output_type=JudgeFeedback,
    )


judge_user_prompt_template = """
Evaluate the agent's performance based on the following criteria:
<CRITERIA>
{criteria}
</CRITERIA>

The agent's final output was:
<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

Tool calls:
<TOOL_CALLS>
{tool_calls}
</TOOL_CALLS>
""".strip()


async def evaluate_agent_performance(result, criteria: list) -> JudgeFeedback:
    """Grade one agent run against natural-language criteria (no assertions)."""
    tool_calls = collect_tools(result.all_messages())

    output = result.output
    if hasattr(output, "model_dump_json"):
        output = output.model_dump_json()

    judge_agent = create_judge_agent()
    judge_user_prompt = judge_user_prompt_template.format(
        criteria="\n".join(criteria),
        output=output,
        tool_calls="\n".join(str(tool_call) for tool_call in tool_calls),
    )

    judge_result = await judge_agent.run(judge_user_prompt)

    feedback = judge_result.output
    print("judge feedback:")
    print(feedback.feedback)
    for criterion in feedback.criteria:
        print(f"  [{criterion.passed}] {criterion.criterion_description}")
        print(f"     {criterion.judgement}")

    return feedback


async def assert_criteria(result, criteria: list) -> JudgeFeedback:
    """Grade one agent run and fail the test if any criterion did not pass."""
    feedback = await evaluate_agent_performance(result, criteria)

    for criterion in feedback.criteria:
        assert criterion.passed, \
            f"{criterion.criterion_description}: {criterion.judgement}"

    return feedback
