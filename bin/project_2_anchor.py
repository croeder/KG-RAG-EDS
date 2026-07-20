#!/usr/bin/env python3
"""Anchor step: resolve a question to a graph node (README build seq #2).

The anchor is entity-linking — get from the words of a question to the node the
question is *about* (the EDS disease node), which is where the traverse then
starts. The question's words won't literally match a node's label, so we match
by meaning: embed the question with the same all-MiniLM model stage 1 used, and
take the nearest node(s) by cosine similarity. Same embedder, same vector space,
same DuckDB cosine function as stage 1 retrieval — the difference is intent. In
stage 1 the nearest node text WAS the answer; here it is only the starting point.

This is currently an exploration, not a committed pipeline step: it prints the
top-k nearest nodes WITH their category so we can see the real risk directly —
635 of 832 nodes are PhenotypicFeature and only 85 are Disease, so the single
nearest hit may be a phenotype (a bad anchor to traverse from) rather than the
disease node. Run with a question as args, or no args for the test spread.
"""

import pathlib
import sys

import duckdb
from sentence_transformers import SentenceTransformer

PROJECT_HOME = str(pathlib.Path(__file__).resolve().parents[1])
DB = f"{PROJECT_HOME}/data/eds.duckdb"

EMBED_MODEL = "all-MiniLM-L6-v2"  # must match what embed_nodes.py used
DIM = 384
K = 5  # how many candidate anchor nodes to show

TEST_QUESTIONS = [
    "What genes cause EDS?",
    "What are the symptoms of Ehlers-Danlos syndrome?",
    "How is the stretchy-skin condition inherited?",
    "What are the types of EDS?",
    "hypermobility disorder",
]


def anchor(con, embedder, question, k=K):
    """Return the k nearest nodes as (id, category, text, similarity)."""
    qv = embedder.encode(question).tolist()
    return con.execute(
        """
        SELECT id, category, text,
               array_cosine_similarity(embedding, ?::FLOAT[384]) AS sim
        FROM nodes
        ORDER BY sim DESC
        LIMIT ?
        """,
        [qv, k],
    ).fetchall()


def main():
    con = duckdb.connect(DB, read_only=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else TEST_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        for id_, category, text, sim in anchor(con, embedder, question):
            cat = category.replace("biolink:", "")
            print(f"  {sim:.3f}  {cat:18} {id_:14} {text[:55]}")


if __name__ == "__main__":
    main()
