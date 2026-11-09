"""Capstone Q2: labeling tool for the D&D agent's evaluation results.

    uv run streamlit run evals/label_app.py

Adapted from the course labeling app (documentation-agent/evals/label_evals.py)
for this project: shows the question, the agent's full response and the tool
calls it made, lets you label good/bad, pick a failure category for bad
responses, add comments, and saves everything to evals/labels.csv so no work
is lost between sessions.
"""

import csv
from pathlib import Path

import streamlit as st

EVALS = Path(__file__).parent
RESULTS = EVALS / 'results.json'
LABELS = EVALS / 'labels.csv'

FAILURE_CATEGORIES = [
    'hallucination',        # invented rules, dice, monsters or book content
    'incorrect-detail',     # wrong number/duration/from a real passage
    'wrong-scope',          # answered a different question, or out-of-scope content
    'incomplete',           # correct but missing a needed part of the answer
    'tool-misuse',          # did not search/read when it had to, or wrong query
    'refused-when-able',    # said it cannot answer though the KB has it
    'notes-error',          # save_note did not store what was asked
]

LABEL_COLUMNS = ['index', 'question', 'label', 'failure_category', 'comments']


def load_results() -> list:
    import json
    with open(RESULTS) as f:
        return json.load(f)


def load_labels() -> dict:
    """index -> label row (saved work wins over session state)."""
    if not LABELS.exists():
        return {}
    with open(LABELS) as f:
        return {int(r['index']): r for r in csv.DictReader(f)}


def save_labels(rows: dict) -> None:
    ordered = [rows[i] for i in sorted(rows)]
    with open(LABELS, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)


st.set_page_config(page_title="D&D Eval Labeling", layout="wide")

results = load_results()
saved = load_labels()

st.sidebar.header("🧭 Navigation")
marked = {i: saved[i]['label'] for i in saved}
selection = st.sidebar.selectbox(
    "Scenario",
    range(len(results)),
    format_func=lambda i: (
        f"{'✅' if marked.get(i) == 'good' else '❌' if marked.get(i) == 'bad' else '⬜'} "
        f"{i + 1}: {results[i]['question'][:35]}..."
    ),
    key="scenario_picker",
)
st.sidebar.progress((len(marked) / len(results)) if results else 0)
st.sidebar.text(f"Labeled: {len(marked)} / {len(results)}")

item = results[selection]
row = saved.get(selection, {})

st.title(f"Scenario {selection + 1} / {len(results)}")

col_meta1, col_meta2, col_meta3 = st.columns(3)
col_meta1.metric("Category", item['category'])
col_meta2.metric("Type", item['type'])
col_meta3.metric("Cost", f"${item['cost']:.5f}")

st.subheader("❓ Question")
st.chat_message("user").write(item['question'])

st.subheader("📝 Agent response")
with st.chat_message("assistant"):
    st.markdown(item['output'])

st.subheader("🛠️ Tool calls")
with st.expander(f"View tool calls ({len(item['tools'])})"):
    for t in item['tools']:
        st.markdown(f"**{t['name']}**")
        st.code(str(t['args']), language="json")

st.divider()
st.subheader("⚖️ Labeling decision")

current = row.get('label', '')
col_a, col_b = st.columns([1, 2])
with col_a:
    decision = st.radio(
        "Is this response good?",
        ["Not selected", "Good", "Bad"],
        index={"good": 1, "bad": 2}.get(current, 0),
        key=f"decision_{selection}",
    )
with col_b:
    category = st.selectbox(
        "Failure category (for bad responses)",
        [""] + FAILURE_CATEGORIES,
        index=([""] + FAILURE_CATEGORIES).index(row.get('failure_category', '') or '')
        if (row.get('failure_category', '') or '') in FAILURE_CATEGORIES else 0,
        key=f"category_{selection}",
    )
    comments = st.text_area(
        "Comments",
        value=row.get('comments', ''),
        height=90,
        key=f"comments_{selection}",
    )

if st.button("💾 Save label & next", type="primary"):
    if decision == "Not selected":
        st.warning("Pick Good or Bad first.")
    elif decision == "Bad" and not category:
        st.warning("Bad responses need a failure category.")
    else:
        rows = load_labels()
        rows[selection] = {
            'index': selection,
            'question': item['question'],
            'label': decision.lower(),
            'failure_category': category if decision == "Bad" else '',
            'comments': comments,
        }
        save_labels(rows)
        next_idx = next(
            (i for i in range(len(results)) if str(i) not in {str(k) for k in rows}),
            None,
        )
        if next_idx is not None:
            st.session_state.scenario_picker = next_idx
        st.rerun()
