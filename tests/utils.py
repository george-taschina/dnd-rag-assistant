"""Shared helpers for the agent tests - same shape as the course's tests/utils.py."""

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]


def _parse_args(args):
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
    return args


def collect_tools(messages) -> list:
    """Real tool calls of a run (PydanticAI's internal final_result excluded)."""
    tool_calls = []

    for message in messages:
        for part in message.parts:
            if part.part_kind != "tool-call":
                continue
            if part.tool_name == "final_result":
                continue
            tool_calls.append(ToolCall(part.tool_name, _parse_args(part.args)))

    return tool_calls


def get_model_name(agent) -> str:
    return f"{agent.model.system}:{agent.model.model_name}"


async def run_agent_test(agent, user_prompt, message_history=None):
    """Run the agent under test and return the result."""
    return await agent.run(user_prompt, message_history=message_history)
