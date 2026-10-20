"""Entry point for the D&D 5E rules agent.

    uv run python main.py "How does an opportunity attack work?"
    uv run python main.py            # runs the default demo question

Loads the knowledge base, builds the agent and runs one question, printing
each tool call followed by the final answer.
"""

import asyncio
import sys

from dotenv import load_dotenv

from doc_agent import DndAgentConfig, create_agent, run_agent
from tools import create_dnd_tools

load_dotenv()

DEFAULT_QUESTION = (
    "How does an opportunity attack work? What triggers it and what does it cost?"
)


async def main(question: str) -> None:
    tools = create_dnd_tools()
    agent = create_agent(DndAgentConfig(), tools)
    await run_agent(agent, question)


def cli() -> None:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    asyncio.run(main(question))


if __name__ == "__main__":
    cli()
