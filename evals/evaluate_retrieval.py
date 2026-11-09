"""Retrieval evaluation across chunking & field-boosting configurations.

    uv run python evals/evaluate_retrieval.py

Implements the course's "Evaluate Retrieval Before Generation" methodology:
- 30 ground-truth D&D queries mapped to expected book filenames and key terms
  that must appear in the retrieved passage.
- Compares 4 search configurations:
    1. chunk(1500, 750) + default boost (1.0)
    2. chunk(3000, 1500) + uniform boost (1.0)
    3. chunk(3000, 1500) + boosted fields (title=3.0, description=2.0, content=1.0) [production]
    4. chunk(6000, 3000) + boosted fields
- Reports Document Hit Rate@3, Document MRR@3, Passage Term Hit Rate@3, and
  average latency per query, then saves results to evals/retrieval_eval.json.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gitsource import chunk_documents
from minsearch import Index

from tools import BOOST, EXCLUDE_FROM_CHUNKING, load_documents, merge_chunks

EVALS = Path(__file__).parent
OUT_FILE = EVALS / "retrieval_eval.json"

PHB = "D&D 5E - Player's Handbook.pdf"
DMG = "D&D 5E - Dungeon Master's Guide.pdf"
MM = "D&D 5E - Monster Manual.pdf"
EBERRON = "D&D 5E - Eberron - Rising from the Last War.pdf"
XGTE = "D&D 5E - Xanathar's Guide to Everything.pdf"
TASHA = "D&D 5E - Tasha's Cauldron of Everything.pdf"
VOLO = "D&D 5E - Volo's Guide to Monsters.pdf"
MTOF = "D&D 5E - Mordenkainen's Tome of Foes.pdf"
DIARIO = "5208a1d4-62b5-495b-8905-382e6f8f946d.pdf"
SCHEDE = "Trascrizione Schede Personaggi D&D.pdf"

CASES = [
    {"query": "opportunity attack reaction reach", "relevant_ids": {PHB}, "term": "opportunity attack"},
    {"query": "fireball components bat guano sulfur", "relevant_ids": {PHB}, "term": "guano"},
    {"query": "concentration saving throw constitution damage", "relevant_ids": {PHB}, "term": "concentration"},
    {"query": "long rest regain spent hit dice half", "relevant_ids": {PHB}, "term": "hit dice"},
    {"query": "cleric starting equipment mace warhammer", "relevant_ids": {PHB}, "term": "warhammer"},
    {"query": "death saving throws critical hit two failures", "relevant_ids": {PHB}, "term": "death saving"},
    {"query": "shield spell magic missile +5 AC", "relevant_ids": {PHB}, "term": "magic missile"},
    {"query": "rogue sneak attack once per turn finesse", "relevant_ids": {PHB}, "term": "sneak attack"},
    {"query": "darkvision 60 feet shades of gray", "relevant_ids": {PHB}, "term": "darkvision"},
    {"query": "grappling athletics acrobatics free hand", "relevant_ids": {PHB}, "term": "grappl"},
    {"query": "optional rule flanking on hexes squares", "relevant_ids": {DMG}, "term": "flanking"},
    {"query": "sword of life stealing rare attunement", "relevant_ids": {DMG}, "term": "life stealing"},
    {"query": "potion of healing drinking action", "relevant_ids": {DMG, PHB}, "term": "potion"},
    {"query": "attunement three magic items short rest", "relevant_ids": {DMG}, "term": "attunement"},
    {"query": "tarrasque hit points 676 gargantuan", "relevant_ids": {MM}, "term": "tarrasque"},
    {"query": "beholder eye rays antimagic cone", "relevant_ids": {MM}, "term": "beholder"},
    {"query": "mind flayer tentacle extract brain", "relevant_ids": {MM, VOLO}, "term": "mind flayer"},
    {"query": "young red dragon fire breath weapon", "relevant_ids": {MM}, "term": "red dragon"},
    {"query": "warforged house cannith last war construct", "relevant_ids": {EBERRON}, "term": "warforged"},
    {"query": "dragonmarked houses Khorvaire Sharn", "relevant_ids": {EBERRON}, "term": "khorvaire"},
    {"query": "artificer infusions eldritch cannon", "relevant_ids": {EBERRON, TASHA}, "term": "artificer"},
    {"query": "oath of conquest paladin guided strike", "relevant_ids": {XGTE}, "term": "conquest"},
    {"query": "hexblade warlock hex warrior", "relevant_ids": {XGTE}, "term": "hexblade"},
    {"query": "gloom stalker ranger dread ambusher", "relevant_ids": {XGTE}, "term": "gloom stalker"},
    {"query": "path of wild magic barbarian wild surge", "relevant_ids": {TASHA}, "term": "wild magic"},
    {"query": "deft explorer favored foe ranger", "relevant_ids": {TASHA}, "term": "deft explorer"},
    {"query": "twilight domain cleric eyes of night", "relevant_ids": {TASHA}, "term": "twilight"},
    {"query": "tabaxi feline agility yuan-ti pureblood", "relevant_ids": {VOLO}, "term": "tabaxi"},
    {"query": "blood war demons devils githyanki", "relevant_ids": {MTOF}, "term": "blood war"},
    {"query": "Diario di Campagna Eberron sessione", "relevant_ids": {DIARIO}, "term": "eberron"},
]


def make_chunks(documents: list, size: int, step: int) -> list:
    out = []
    for doc in documents:
        if doc["filename"] in EXCLUDE_FROM_CHUNKING:
            out.append({**doc, "start": 0, "chunk_id": 0})
        else:
            for i, ch in enumerate(chunk_documents([doc], size=size, step=step)):
                ch["chunk_id"] = i
                out.append(ch)
    return out


def search_with_boost(index: Index, query: str, boost_dict: dict, k: int = 3) -> list:
    hits = index.search(query, num_results=k * 3, boost_dict=boost_dict)
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
    return passages[:k]


def retrieval_metrics(cases: list, results_per_case: list, k: int = 3) -> dict:
    hits = []
    reciprocal_ranks = []
    term_hits = []
    total_chars = 0

    for case, passages in zip(cases, results_per_case, strict=True):
        relevant = case["relevant_ids"]
        ranked_ids = [p["filename"] for p in passages[:k]]
        first_rank = next(
            (rank for rank, doc_id in enumerate(ranked_ids, 1) if doc_id in relevant),
            None,
        )
        hits.append(int(first_rank is not None))
        reciprocal_ranks.append(1 / first_rank if first_rank else 0.0)

        combined_text = " ".join(p["content"].lower() for p in passages[:k])
        term_hits.append(int(case["term"].lower() in combined_text))
        total_chars += sum(len(p["content"]) for p in passages[:k])

    n = len(cases)
    return {
        "doc_hit_rate_at_3": round(sum(hits) / n, 4),
        "doc_mrr_at_3": round(sum(reciprocal_ranks) / n, 4),
        "term_hit_rate_at_3": round(sum(term_hits) / n, 4),
        "avg_passage_chars": round(total_chars / n),
    }


def main() -> None:
    documents = load_documents()
    configs = [
        {"name": "chunk_1500_step_750_title_heavy", "size": 1500, "step": 750, "boost": {"title": 3.0, "description": 2.0, "content": 1.0}},
        {"name": "chunk_3000_step_1500_title_heavy (baseline)", "size": 3000, "step": 1500, "boost": {"title": 3.0, "description": 2.0, "content": 1.0}},
        {"name": "chunk_3000_step_1500_uniform", "size": 3000, "step": 1500, "boost": {"title": 1.0, "description": 1.0, "content": 1.0}},
        {"name": "chunk_3000_step_1500_content_heavy (selected)", "size": 3000, "step": 1500, "boost": BOOST},
        {"name": "chunk_6000_step_3000_content_heavy", "size": 6000, "step": 3000, "boost": BOOST},
    ]

    summary = []
    by_size = {}
    for cfg in configs:
        key = (cfg["size"], cfg["step"])
        if key not in by_size:
            chunks = make_chunks(documents, cfg["size"], cfg["step"])
            idx = Index(text_fields=["title", "description", "content"], keyword_fields=["filename"])
            idx.fit(chunks)
            by_size[key] = (idx, len(chunks))
        idx, num_chunks = by_size[key]

        t0 = time.time()
        all_results = [search_with_boost(idx, c["query"], cfg["boost"], k=3) for c in CASES]
        avg_ms = round((time.time() - t0) * 1000 / len(CASES), 2)

        metrics = retrieval_metrics(CASES, all_results, k=3)
        row = {
            "config": cfg["name"],
            "chunk_size": cfg["size"],
            "chunk_step": cfg["step"],
            "num_chunks": num_chunks,
            "boost": cfg["boost"],
            **metrics,
            "avg_latency_ms": avg_ms,
        }
        summary.append(row)
        print(
            f"{cfg['name']:40s} | doc_hit@3={metrics['doc_hit_rate_at_3']:.3f} "
            f"| mrr@3={metrics['doc_mrr_at_3']:.3f} "
            f"| term_hit@3={metrics['term_hit_rate_at_3']:.3f} "
            f"| chars={metrics['avg_passage_chars']:5d} | {avg_ms:5.1f}ms"
        )

    OUT_FILE.write_text(json.dumps({"num_cases": len(CASES), "configurations": summary}, indent=2))
    print(f"\nSaved retrieval evaluation to {OUT_FILE}")


if __name__ == "__main__":
    main()
