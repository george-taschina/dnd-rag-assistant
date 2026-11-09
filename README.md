# D&D 5E Rules & Character Assistant

A RAG assistant for Dungeons & Dragons 5th Edition that answers rules, spell and character questions directly from the official rulebooks, and cites the source it took each answer from.

## The Problem

D&D players and DMs need to look up rules mid-session — how opportunity attacks work, which spells a 3rd-level wizard can prepare, what a monster's action economy looks like. The answer is always in one of the 5e rulebooks, but the books are 500+ pages of PDF: searching them by hand is slow, and answers from general-purpose LLMs are often confident but wrong, with no way to check where they came from.

## What It Does

The user asks a question in plain language (English or Italian). The system:

1. retrieves the most relevant passages from the rulebooks it has indexed (Player's Handbook, Dungeon Master's Guide, Monster Manual, Eberron, Xanathar's, Tasha's, Volo's, Mordenkainen's, plus the campaign's own notes),
2. builds a grounded prompt from those passages,
3. returns an answer that cites the exact book (and, where useful, the passage) it came from — or says it doesn't know when the context doesn't contain the answer.

A typical interaction: *"What are the rules for opportunity attacks?"* → an answer quoting the relevant Player's Handbook passage, attributed to **Player's Handbook (D&D 5E)**.

Since week 3 the pipeline is an **agent with tools**: instead of always doing exactly one search, the model decides what to do — search the index, read a longer window of a book, list what it knows, or write a campaign note down — and keeps going until it can answer.

## Setup

1. Install uv if you don't have it yet: https://docs.astral.sh/uv/getting-started/installation/

2. Clone this repository (or download the zip and extract it).

3. Create a `.env` file from the template and add your API key:

       cp .env.example .env

4. Install dependencies:

       uv sync

5. Start Jupyter:

       uv run jupyter notebook

## Notebooks

- `notebooks/01-setup.ipynb` - smoke test that confirms your environment works
- `notebooks/02-rag.ipynb` - a minimal RAG baseline you can adapt to your own data
- `notebooks/03-agent.ipynb` - week 3: the same pipeline turned into an agent with tools

## Project Layout

Notebooks are great for exploring, but they are hard to import and hard to test. The week‑3
agent is therefore also available as a small Python project (the three‑module layout from the
*Testing AI Applications* course):

| file | contents |
|---|---|
| `tools.py` | knowledge‑base loading (`load_documents`, `chunk_documents_cached`, `build_index`), `merge_chunks`/`search_passages`, and the `DndTools` class with the four tools + `create_dnd_tools()` |
| `doc_agent.py` | `DEFAULT_INSTRUCTIONS`, `DndAgentConfig`, `create_agent()`, the `NamedCallback` that prints every tool call, and `run_agent()` |
| `main.py` | entry point: `uv run python main.py "your question"` |

Run the agent from the command line:

    uv run python main.py "Which spells can a 3rd-level wizard prepare?"

    QUESTION: Which spells can a 3rd-level wizard prepare?
    TOOL CALL (dnd-rules-agent): search_rules({"query":"3rd level wizard prepared spells"})
    TOOL CALL (dnd-rules-agent): read_passage({"filename":"D&D 5E - Player's Handbook.pdf","start":148500,"length":4000})
    ANSWER: ...

The chunked index is cached in `.cache/chunks.pkl` (gitignored) so a run starts instantly
after the first one — the same idea as `create_documentation_tools_cached` in the course
reference project.

## The Agent (week 3)

`notebooks/03-agent.ipynb` keeps the week‑1 data, index and citations, and replaces the fixed
`search → build_prompt → llm → rag` chain with a PydanticAI agent (OpenAI `gpt-4o-mini`, Responses API) that chooses between four small tools:

| tool | what it does |
|---|---|
| `search_rules(query, num_results=3)` | the week‑1 `search` as a tool: passages with book, filename and character offset |
| `read_passage(filename, start, length)` | read a longer window of one book when a passage stops mid‑rule |
| `list_sources()` | the books in the knowledge base — answers "what do you know?" and scopes the answer |
| `save_note(topic, note)` | write a house rule / campaign note to `data/notes.json` so later sessions know it |

Every run prints the tool calls with their arguments and results, so a wrong answer can be traced to the exact call that misled it.

## Monitoring (week 5)

Week 5 capstone: the agent is instrumented with **Pydantic Logfire** and wrapped in a
Streamlit chat app.

| file | what it does |
|---|---|
| `monitoring.py` | `logfire.configure` + `logfire.instrument_pydantic_ai()`, the `dnd_session` span helper (one browser session = one trace tree) and the shared event loop used by the app |
| `cost_tracker.py` | per-session token/cost accumulation (course `cost_tracker` pattern, gpt-4o-mini pricing) |
| `app.py` | Streamlit chat: per-session trace grouping via `logfire.attach_context`, tool-call expander, live session usage/cost in the sidebar |
| `check_trace.py` | runs one query, then reads the trace back from Logfire with the query client and reconstructs the run with `trace_replay` (course `otel_test.py` pattern) |
| `collect_data.py` | drives `app.py` with `streamlit.testing.v1.AppTest` to collect real interactions and summarise them from Logfire → `data/usage_collection.json` |
| `trace_replay/` | course module: OTel trace → PydanticAI `AgentRunResult` (patched for the current query client and pydantic-ai 2.x usage attributes) |

```bash
uv run streamlit run app.py     # chat UI, one Logfire trace tree per browser session
uv run python check_trace.py    # run a query and inspect its trace
uv run python collect_data.py   # collect usage data through the app
```

The dashboard (`george-taschina/starter-project`) shows for every session:
`dnd_session` → `invoke_agent dnd-rules-agent` → `chat gpt-4o-mini` LLM calls and
`execute_tool search_rules / read_passage / list_sources / save_note` spans, with
token usage (`gen_ai.aggregated_usage`) and cost per span.

`.env` needs `OPENAI_API_KEY`, `LOGFIRE_WRITE_TOKEN` and `LOGFIRE_READ_TOKEN`
(project settings → write/read tokens).

## Evaluation (week 6)

Week 6 capstone: offline evaluation pipeline (`evals/`) — 50 scenarios across five categories, human labeling tool, and an aligned LLM judge.

| file | what it does |
|---|---|
| `evals/scenarios.csv` | 50 D&D-specific evaluation scenarios (15 `happy_path`, 10 `varied_phrasing`, 10 `edge_case`, 7 `out_of_scope`, 8 `breaking`) |
| `evals/run_batch.py` | runs the agent on all 50 scenarios in batch → `evals/results.json` ($0.2064, 584s) |
| `evals/label_app.py` | Streamlit labeling app (adapted from the course reference): shows question, full response and tool calls, good/bad + failure-category dropdown, saves to `evals/labels.csv` |
| `evals/labels.csv` | human gold labels for all 50 scenarios (**43 good / 7 bad**; failure categories: `incorrect-detail` (3), `hallucination` (2), `wrong-scope` (1), `incomplete` (1)) |
| `evals/judge.py` | PydanticAI structured-output LLM judge (`reasoning` + `label`), refined after analyzing v1 disagreements |
| `evals/results_judged_v1.json`, `evals/results_judged.json` | judge outputs before (v1) and after (v2) prompt improvement |
| `evals/metrics.py` | computes TP/FP/FN/TN, accuracy, precision, recall, and prints every disagreement |

| metric | judge v1 | judge v2 (improved) |
|---|---|---|
| accuracy | 0.86 (43/50) | **0.92 (46/50)** |
| precision ("bad") | 0.50 (5/10) | **0.67 (6/9)** |
| recall ("bad") | 0.71 (5/7) | **0.86 (6/7)** |
| disagreements | 7 | **4** |

```bash
uv run python evals/run_batch.py
uv run streamlit run evals/label_app.py
uv run python evals/judge.py
uv run python evals/metrics.py evals/results_judged.json
```

## Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/test_judge.py -s     # print the judge's reasoning
```

| file | what it covers |
|---|---|
| `tests/conftest.py` | module-scoped `agent` fixture: index + agent built once per run |
| `tests/utils.py` | `collect_tools` (tool calls out of a run's messages), `run_agent_test` |
| `tests/test_agent.py` | happy path + tool order + citation, out-of-scope refusal, `list_sources` scope question |
| `tests/judge.py` | the LLM judge: `evaluate_agent_performance` and `assert_criteria` |
| `tests/test_judge.py` | three natural-language criteria for the opportunity-attack answer |

pytest and pytest-asyncio run in the dev dependency group; `asyncio_default_*_loop_scope = "module"` keeps the fixture's HTTP client on the same event loop as the tests.

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.

The D&D 5E rulebooks (PDF) live in `data/pdfs/`. The first run of `notebooks/02-rag.ipynb` extracts their text into `data/documents.json`; later runs load that cache and start instantly.

`data/notes.json` is the agent's memory: it is written by the `save_note` tool (the notebook resets it before the demo so re-running doesn't duplicate entries).
