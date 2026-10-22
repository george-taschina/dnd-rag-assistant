"""Streamlit chat app for the D&D 5E rules agent (monitoring capstone Q2).

    uv run streamlit run app.py

Every browser session is one Logfire trace tree (`dnd_session` span); every
question is an agent run attached to that tree, so the dashboard shows the
whole conversation, each tool call, and the session's token usage and cost.
"""

import uuid

import logfire
import streamlit as st

import monitoring  # noqa: F401  (configures + instruments Logfire)
from cost_tracker import SessionCost
from doc_agent import (
    DndAgentConfig,
    create_agent,
    format_tool_calls_from_messages,
)
from tools import create_dnd_tools

st.set_page_config(page_title="D&D 5E Rules Agent", page_icon="🐉", layout="wide")

st.title("🐉 D&D 5E Rules Agent")
st.caption(
    "Answers rules, spell and character questions from the official "
    "rulebooks, with citations. Monitored with Pydantic Logfire."
)


@st.cache_resource
def get_agent():
    """Build the agent once per server process (the index load takes seconds)."""
    return create_agent(DndAgentConfig(), create_dnd_tools())


agent = get_agent()

# ── Session state: one browser session = one trace tree ─────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []   # display entries: {role, content, tools}
    st.session_state.history = []    # raw pydantic-ai messages for the agent
    st.session_state.cost = SessionCost()
    # lesson pattern: open the session span once, save its context, and
    # re-attach it on every rerun so all runs share one trace tree
    with logfire.span("dnd_session", session_id=st.session_state.session_id):
        st.session_state.lf_ctx = logfire.get_context()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            with st.expander(f"🔧 Tools called ({len(msg['tools'])})"):
                for t in msg["tools"]:
                    st.code(f"{t['name']}({t['args']})", language="python")

# ── Sidebar: live session usage ─────────────────────────────────────────────
cost: SessionCost = st.session_state.cost
st.sidebar.markdown("### Session usage")
st.sidebar.markdown(
    f"""
| metric | value |
| --- | --- |
| session id | `{st.session_state.session_id}` |
| questions | {cost.runs} |
| input tokens | {cost.input_tokens:,} |
| output tokens | {cost.output_tokens:,} |
| session cost | ${cost.cost_usd:.6f} |
"""
)
st.sidebar.markdown(
    f"[Open traces](https://logfire-eu.pydantic.dev/george-taschina/starter-project/traces)"
)

# ── Chat loop ───────────────────────────────────────────────────────────────
question = st.chat_input("Ask a D&D 5E rules question…")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Consulting the rulebooks…"):
            history = st.session_state.history
            with logfire.attach_context(st.session_state.lf_ctx):
                result = monitoring.run_async(
                    agent.run(question, message_history=history)
                )
        st.markdown(result.output)

        # only THIS turn's tool calls - all_messages() would also count the history
        tools = format_tool_calls_from_messages(result.all_messages()[len(history):])
        if tools:
            with st.expander(f"🔧 Tools called ({len(tools)})"):
                for t in tools:
                    st.code(f"{t['name']}({t['args']})", language="python")

    # keep only the NEW raw messages so the next turn's history is not duplicated
    st.session_state.history.extend(result.all_messages()[len(history):])
    st.session_state.messages.append(
        {"role": "assistant", "content": result.output, "tools": tools}
    )

    cost.add(result.usage)
    st.rerun()
