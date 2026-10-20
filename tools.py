"""Search tools and knowledge-base loading for the D&D 5E rules agent.

This is `notebooks/03-agent.ipynb` converted into a module, following the
course's project layout (tools.py = search tools + data loading, doc_agent.py
= agent configuration, main.py = entry point).

The knowledge base is the 10 D&D 5E PDFs in ``data/pdfs/``: their text is
cached in ``data/documents.json`` (produced by ``notebooks/02-rag.ipynb``),
chunked with an overlap, and indexed with minsearch.
"""

import json
import pickle
from pathlib import Path

from gitsource import chunk_documents
from minsearch import Index

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"

DOCUMENTS_CACHE = DATA_DIR / "documents.json"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"
NOTES_FILE = DATA_DIR / "notes.json"

# Small enough to index as a single document.
EXCLUDE_FROM_CHUNKING = {
    "5208a1d4-62b5-495b-8905-382e6f8f946d.pdf",
    "Trascrizione Schede Personaggi D&D.pdf",
}

BOOST = {"title": 3.0, "description": 2.0, "content": 1.0}


def load_documents() -> list:
    """Load the cached PDF text produced by notebooks/02-rag.ipynb."""
    if not DOCUMENTS_CACHE.exists():
        raise FileNotFoundError(
            f"{DOCUMENTS_CACHE} not found - run notebooks/02-rag.ipynb once to "
            "extract the PDFs."
        )
    return json.loads(DOCUMENTS_CACHE.read_text())


def chunk_documents_cached(documents: list) -> list:
    """Split every book into overlapping chunks, with a pickle cache.

    Chunking 500MB of books takes a few seconds; tests want it instant.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    if CHUNKS_CACHE.exists():
        with open(CHUNKS_CACHE, "rb") as f:
            return pickle.load(f)

    document_chunks = []
    for doc in documents:
        if doc["filename"] in EXCLUDE_FROM_CHUNKING:
            document_chunks.append({**doc, "start": 0, "chunk_id": 0})
        else:
            for i, chunk in enumerate(chunk_documents([doc], size=3000, step=1500)):
                chunk["chunk_id"] = i
                document_chunks.append(chunk)

    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(document_chunks, f)

    return document_chunks


def build_index(document_chunks: list) -> Index:
    """Fit the minsearch index over the chunked books."""
    index = Index(
        text_fields=["title", "description", "content"],
        keyword_fields=["filename"],
    )
    index.fit(document_chunks)
    return index


def merge_chunks(chunks: list) -> str:
    """Merge overlapping/adjacent chunks from the same document into one passage.

    Chunks know their absolute 'start' offset in the original document, so the
    overlap is written only once and the result is exact contiguous text.
    """
    chunks = sorted(chunks, key=lambda c: c["start"])
    merged = chunks[0]["content"]
    end = chunks[0]["start"] + len(merged)

    for chunk in chunks[1:]:
        chunk_end = chunk["start"] + len(chunk["content"])
        if chunk_end <= end:
            continue  # fully contained in what we already have
        overlap = max(end - chunk["start"], 0)
        if overlap == 0:
            merged += "\n...\n"  # gap between non-adjacent chunks
        merged += chunk["content"][overlap:]
        end = chunk_end

    return merged


def search_passages(index: Index, query: str, num_results: int = 3) -> list:
    """Search chunks, then merge adjacent chunks per book into passages.

    Every passage carries the book title, the source filename and the
    character offset ``start`` of its first chunk, so ``read_passage`` can
    widen the window around a hit.
    """
    # Fetch extra raw chunks: adjacent hits from the same book collapse
    # into a single merged passage.
    hits = index.search(query, num_results=num_results * 3, boost_dict=BOOST)

    grouped = {}
    for hit in hits:
        grouped.setdefault(hit["filename"], []).append(hit)

    passages = []
    for filename, chunks in grouped.items():
        chunks = sorted(chunks, key=lambda c: c["start"])
        content = merge_chunks(chunks) if len(chunks) > 1 else chunks[0]["content"]
        passages.append(
            {
                "title": chunks[0]["title"],
                "filename": filename,
                "start": chunks[0]["start"],
                "content": content,
            }
        )

    return passages[:num_results]


class DndTools:
    """The four tools the agent can call (see notebooks/03-agent.ipynb)."""

    def __init__(self, index: Index, documents: list) -> None:
        self.index = index
        self.documents = documents
        self.by_filename = {d["filename"]: d for d in documents}

    # --- read/search -----------------------------------------------------

    def search_rules(self, query: str, num_results: int = 3) -> list:
        """Search the D&D 5E rulebook index for a rules, spell, monster or
        character question. Returns passages: book `title`, `filename`, the
        character offset `start` in that book, and the passage `content`.
        Call this first for any question about the rules."""
        return search_passages(self.index, query, num_results=num_results)

    def read_passage(self, filename: str, start: int, length: int = 4000) -> str:
        """Read `length` characters of one book starting at character offset
        `start`. Use it when a search passage cuts off in the middle of a rule.
        `filename` and `start` must come from a search_rules result."""
        doc = self.by_filename.get(filename)
        if doc is None:
            return f"error: no book named '{filename}' - call list_sources first"
        return doc["content"][start:start + length]

    # --- scope and memory ------------------------------------------------

    def list_sources(self) -> list:
        """List the books in the knowledge base (title, filename, size). Use it
        to answer 'what do you know about' questions and to check scope."""
        return [
            {"title": d["title"], "filename": d["filename"], "characters": len(d["content"])}
            for d in self.documents
        ]

    def save_note(self, topic: str, note: str) -> str:
        """Save a campaign note or house rule to data/notes.json so that later
        sessions know about it. Use when the user asks you to remember, note
        down or record something."""
        notes = json.loads(NOTES_FILE.read_text()) if NOTES_FILE.exists() else []
        notes.append({"topic": topic, "note": note})
        NOTES_FILE.write_text(json.dumps(notes, indent=2, ensure_ascii=False))
        return f"saved note #{len(notes)} '{topic}' to {NOTES_FILE.relative_to(ROOT)}"


def create_dnd_tools() -> DndTools:
    """Load the knowledge base (cached chunks if available) and return the tools."""
    documents = load_documents()
    index = build_index(chunk_documents_cached(documents))
    return DndTools(index, documents)
