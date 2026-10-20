"""Agent configuration and execution for the D&D 5E rules agent.

Import order follows the course convention: stdlib, third-party, project modules.
"""

import json
from dataclasses import dataclass

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import FunctionToolCallEvent

from tools import DndTools

DEFAULT_INSTRUCTIONS = """
You are a Dungeons & Dragons 5th Edition rules assistant with four tools:
search_rules, read_passage, list_sources and save_note. Use the tools; do not
answer rules questions from memory.

How to work:
1. For any rules, spell, monster or character question, start with one
   search_rules call. Reformulate the query once if the first search is empty.
2. If a passage stops in the middle of a rule, use read_passage with that
   passage's filename and start offset to read more of the same book. Do not
   read blindly: only use offsets you got from search_rules.
3. For "which books do you know" / "can you answer X" questions, call
   list_sources before answering, and be honest about what is not in them.
4. When the user asks you to remember, note down or record something
   (usually a house rule or a campaign note), call save_note with a short
   topic and the note.

Answer rules:
- Cite the book for every claim, e.g. (Player's Handbook, D&D 5E), using the
  title of the passage you actually read. Never invent a rule, a number or a
  page that is not in the retrieved text.
- If the tools do not contain the answer, say you don't know and name the book
  that would most likely contain it.
- Answer in the language the user writes in (English or Italian).
- For character-building questions give the concrete rules (prerequisites,
  slots, action economy), not general advice.
- If a question is about another game, or is not about D&D at all, say so in
  one short sentence and offer to answer a D&D question instead. Do not search
  more than once for an obviously unrelated question.
""".strip()


@dataclass
class DndAgentConfig:
    model: str = "openai-responses:gpt-4o-mini"
    name: str = "dnd-rules-agent"
    instructions: str = DEFAULT_INSTRUCTIONS


def create_agent(config: DndAgentConfig, tools: DndTools) -> Agent:
    """Attach the four tools to a configured agent."""
    return Agent(
        config.model,
        name=config.name,
        instructions=config.instructions,
        tools=[
            tools.search_rules,
            tools.read_passage,
            tools.list_sources,
            tools.save_note,
        ],
    )


class NamedCallback:
    """Prints every tool call as it happens, like the course's doc_agent.py."""

    def __init__(self, agent: Agent) -> None:
        self.agent_name = agent.name

    async def print_function_calls(self, ctx, event):
        # Detect nested streams
        if hasattr(event, "__aiter__"):
            async for sub in event:
                await self.print_function_calls(ctx, sub)
            return

        if isinstance(event, FunctionToolCallEvent):
            tool_name = event.part.tool_name
            args = event.part.args
            print(f"TOOL CALL ({self.agent_name}): {tool_name}({args})")

    async def __call__(self, ctx, event):
        return await self.print_function_calls(ctx, event)


async def run_agent(
    agent: Agent,
    question: str,
    message_history=None,
) -> AgentRunResult:
    """Run the agent on one question, printing the tool calls it makes."""
    if message_history is None:
        message_history = []

    callback = NamedCallback(agent)

    print(f"QUESTION: {question}")
    result = await agent.run(
        question,
        message_history=message_history,
        event_stream_handler=callback,
    )

    print(f"ANSWER:\n{result.output}")
    return result


def format_tool_calls(result: AgentRunResult) -> list:
    """Tool calls of a finished run as [{name, args}, ...] (skips final_result).

    PydanticAI hands over the arguments as a JSON string, so they are parsed
    here to keep assertions in tests readable.
    """
    tool_calls = []
    for message in result.all_messages():
        for part in message.parts:
            if part.part_kind != "tool-call" or part.tool_name == "final_result":
                continue
            args = part.args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append({"name": part.tool_name, "args": args})
    return tool_calls