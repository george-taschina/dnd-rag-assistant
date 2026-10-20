"""pytest fixtures: the knowledge base and agent under test, built once."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from doc_agent import DndAgentConfig, create_agent
from tools import create_dnd_tools


@pytest.fixture(scope="module")
def agent():
    """The D&D rules agent: index loaded once per module (it takes a few seconds)."""
    tools = create_dnd_tools()
    return create_agent(DndAgentConfig(), tools)
