#!/usr/bin/env python3
"""Retrieval baseline for projects 1 and 2 against the derived gold (project 4).

Reads eval/gold_eds.resolved.jsonl and, per question, runs both retrieval
pipelines and scores the entities they pull against the gold answer set:

  - Project 1 (plain vector RAG): the top-k nearest node ids (rag_query.retrieve).
  - Project 2 (KG-RAG): the node ids one hop from the disambiguated anchor along
    the top predicate.

To measure the REAL project-2 pipeline without touching it, this composes the
pipeline's own functions (picked_predicates -> anchor -> disambiguate -> traverse)
in the same order project_2_kg_rag_query.main() does. It imports those functions
rather than re-implementing them, so the numbers reflect the shipped code; the
only thing not reused is main()'s printing and its MAX_FACTS generation cap, which
is a generation concern, not a retrieval one (traverse reach is scored uncapped).

Metric is precision / recall / F1 of the pulled entity set vs answer_entities,
per case, then macro-averaged. This is the baseline the design notes insist on
before any knob (cutoff, k, disambiguation) is touched — see
doc/project_4_evaluation.md and eval/README.md.

Read the two systems' numbers differently:
  - Project 2's answer set IS graph edges, so entity overlap is the honest metric.
  - Project 1 retrieves whole node descriptions and leans on the generator to
    extract; its entity recall understates it, because grounding on the EDS
    description is not the same as pulling each phenotype node. Low P1 entity
    recall is expected and should be read next to groundedness, not alone.

Run: venv/bin/python3 bin/eval_score.py
"""

import json
import pathlib

import duckdb
import rag_query
from project_2_anchor import anchor
from project_2_disambiguate import RECALL_K, disambiguate, picked_predicates
from project_2_kg_rag_query import MAX_FACTS
from project_2_predicate_classifier import load_config
from project_2_traverse import traverse
from sentence_transformers import SentenceTransformer

REPO = pathlib.Path(__file__).resolve().parents[1]
DB = REPO / "data" / "eds.duckdb"
GOLD = REPO / "eval" / "gold_eds.resolved.jsonl"


def project2_retrieval(con, model, pred_ids, pred_emb, cutoff, question):
    """Compose the shipped project-2 retrieval steps; return (anchor_node, pred, rows).

    Mirrors project_2_kg_rag_query.main() up to (but not including) generation, so
    the scorer measures the real pipeline. rows are traverse()'s full output — not
    capped at MAX_FACTS, since that cap is for the generator, not retrieval.
    """
    preds = picked_predicates(question, model, pred_ids, pred_emb, cutoff)
    if not preds:
        return None, None, []
    candidates = anchor(con, model, question, k=RECALL_K)
    _rank, anchor_node, _count = disambiguate(con, candidates, preds)
    if anchor_node is None:
        return None, None, []
    top_pred = preds[0]
    return anchor_node, top_pred, traverse(con, anchor_node[0], top_pred)


def prf(retrieved, gold):
    """Precision, recall, F1, and the raw counts for one retrieved-vs-gold pair."""
    r, g = set(retrieved), set(gold)
    tp = len(r & g)
    precision = tp / len(r) if r else 0.0
    recall = tp / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"p": precision, "r": recall, "f1": f1, "tp": tp, "n_ret": len(r), "n_gold": len(g)}


def macro(scores, key):
    return sum(s[key] for s in scores) / len(scores) if scores else 0.0


def main():
    cases = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    con = duckdb.connect(str(DB), read_only=True)

    embed_model, cutoff, predicates = load_config()
    # Both pipelines must share the vector space the nodes were embedded in.
    assert embed_model == rag_query.EMBED_MODEL, (
        f"embed model mismatch: config {embed_model} vs rag_query {rag_query.EMBED_MODEL}"
    )
    model = SentenceTransformer(embed_model)
    pred_ids = list(predicates)
    pred_emb = model.encode(list(predicates.values()), convert_to_tensor=True)

    p1_scores, p2_scores = [], []
    rows_out = []
    for c in cases:
        gold = c["answer_entities"]

        hits = rag_query.retrieve(con, model, c["question"])  # k = rag_query.K
        p1 = prf([h[0] for h in hits], gold)

        anchor_node, pred, rows = project2_retrieval(
            con, model, pred_ids, pred_emb, cutoff, c["question"]
        )
        p2 = prf([r[0] for r in rows], gold)

        p1_scores.append(p1)
        p2_scores.append(p2)
        rows_out.append(
            {
                "id": c["id"],
                "expand": c["expand_subclasses"],
                "p1": p1,
                "p2": p2,
                "anchor": anchor_node[0] if anchor_node else "-",
                "pred": pred.replace("biolink:", "") if pred else "-",
                "trunc": len(rows) > MAX_FACTS,
            }
        )

    hdr = f"{'case':<26} {'exp':>3}  {'P1 P/R/F1':>18}   {'P2 P/R/F1':>18}  P2 anchor/pred"
    print(hdr)
    print("-" * len(hdr))
    for o in rows_out:
        p1, p2 = o["p1"], o["p2"]
        t = "*" if o["trunc"] else " "
        print(
            f"{o['id']:<26} {str(o['expand'])[0]:>3}  "
            f"{p1['p']:.2f}/{p1['r']:.2f}/{p1['f1']:.2f}   "
            f"{p2['p']:.2f}/{p2['r']:.2f}/{p2['f1']:.2f}{t}  "
            f"{o['anchor']} {o['pred']}"
        )

    print("-" * len(hdr))
    print(
        f"{'MACRO (n=' + str(len(cases)) + ')':<26} {'':>3}  "
        f"{macro(p1_scores, 'p'):.2f}/{macro(p1_scores, 'r'):.2f}/{macro(p1_scores, 'f1'):.2f}   "
        f"{macro(p2_scores, 'p'):.2f}/{macro(p2_scores, 'r'):.2f}/{macro(p2_scores, 'f1'):.2f}"
    )
    print("\n* = project-2 facts exceed MAX_FACTS (capped for generation only; retrieval scored uncapped).")
    print("exp = gold aggregates across the subtype subtree; the simple pipeline anchors ONE node.")


if __name__ == "__main__":
    main()
