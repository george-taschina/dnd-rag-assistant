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

## Data

Put your project data in the `data/` folder. See `notebooks/02-rag.ipynb` for how to load it.

The D&D 5E rulebooks (PDF) live in `data/pdfs/`. The first run of `notebooks/02-rag.ipynb` extracts their text into `data/documents.json`; later runs load that cache and start instantly.
