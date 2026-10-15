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

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.

The D&D 5E rulebooks (PDF) live in `data/pdfs/`. The first run of `notebooks/02-rag.ipynb` extracts their text into `data/documents.json`; later runs load that cache and start instantly.

`data/notes.json` is the agent's memory: it is written by the `save_note` tool (the notebook resets it before the demo so re-running doesn't duplicate entries).
