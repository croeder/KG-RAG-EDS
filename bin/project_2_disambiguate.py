#!/usr/bin/env python3
"""Disambiguate the anchor using the picked predicate (README build seq #2).

This is the glue that couples the two entry-point steps. Anchor recall
(project_2_anchor.py) returns the nodes nearest the question by embedding, but
nearest-by-meaning is often a symptom, not the disease we want to traverse from.
The fix is graph-native and deliberately simple: among the candidates, pick the
one with the MOST edges of the picked predicate(s) — the anchor is the node most
connected through the relevant relation. The disease is a hub (100+ has_phenotype
edges) while a leaf phenotype has a handful, so counting swamps the leaves. No
per-predicate config, no direction table, and it holds even when the predicate
set is loose. The predicate constrains the anchor (see doc/project_2_KG_RAG.md).

It is a heuristic: it assumes the anchor is well-connected (true here), and the
right disease still has to be IN the candidate pool — a recall miss defeats any
method — so recall runs with a generous k.

Run with a question as args, or no args for the test set (includes the cases
where the top embedding hit is the wrong node).
"""

import sys

import duckdb
from project_2_anchor import anchor
from project_2_predicate_classifier import classify, load_config
from sentence_transformers import SentenceTransformer

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

RECALL_K = 25  # generous: the disease must be in the candidate pool to be picked

TEST_QUESTIONS = [
    "What genes cause EDS?",
    "What are the symptoms of Ehlers-Danlos syndrome?",
    "How is the stretchy-skin condition inherited?",  # top hits are skin phenotypes
    "hypermobility disorder",  # top hit is the phenotype, not the disease
]


def picked_predicates(question, model, pred_ids, pred_emb, cutoff):
    """Predicates at/above the cutoff, highest similarity first."""
    ranked = classify(question, model, pred_ids, pred_emb)
    return [p for p, score in ranked if score >= cutoff]


def edge_count_of(con, node_id, predicates):
    """How many edges have node_id on either side with one of these predicates."""
    return con.execute(
        """
        SELECT count(*) FROM edges
        WHERE list_contains($preds, predicate)
          AND (subject = $n OR object = $n)
        """,
        {"preds": predicates, "n": node_id},
    ).fetchone()[0]


def disambiguate(con, candidates, predicates):
    """Pick the candidate most connected via the picked predicates (the hub).

    Ties break toward the better embedding rank (strict >, first wins).
    Returns (rank, (id, category, text, sim), count) or (None, None, 0).
    """
    best = (None, None, 0)
    for rank, cand in enumerate(candidates):
        count = edge_count_of(con, cand[0], predicates)
        if count > best[2]:
            best = (rank, cand, count)
    return best


def main():
    con = duckdb.connect(DB, read_only=True)
    embed_model, cutoff, predicates = load_config()
    model = SentenceTransformer(embed_model)
    pred_ids = list(predicates)
    pred_emb = model.encode(list(predicates.values()), convert_to_tensor=True)

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else TEST_QUESTIONS

    for question in questions:
        preds = picked_predicates(question, model, pred_ids, pred_emb, cutoff)
        candidates = anchor(con, model, question, k=RECALL_K)
        rank, chosen, count = disambiguate(con, candidates, preds)

        print(f"\nQ: {question}")
        print(f"   predicates: {[p.replace('biolink:', '') for p in preds]}")
        top = candidates[0]
        print(f"   top embedding hit: {top[1].replace('biolink:', '')} {top[0]} {top[2][:40]}")
        if chosen is None:
            print("   -> no candidate has an edge of these predicates (no usable anchor)")
        else:
            print(
                f"   -> anchor: {chosen[1].replace('biolink:', '')} {chosen[0]} "
                f"{chosen[2][:40]}  (rank {rank} of {RECALL_K}, {count} edges)"
            )


if __name__ == "__main__":
    main()
