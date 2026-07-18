#!/usr/bin/env python3
"""End-to-end KG-RAG query (README build seq #2): graph retrieval + generation.

    question
      -> anchor recall     (embed the question, nearest nodes)      [entry: embed]
      -> predicate pick    (embed vs predicate descriptions)        [entry: embed]
      -> disambiguate      (degree/hub among the candidates)        [graph]
      -> traverse          (SQL, one hop along the top predicate)   [graph]
      -> facts -> generation seam writes the answer.

The two embedding steps are the only places English touches vector space; from
disambiguation onward it is the graph doing the work. Generation is reused
unchanged from stage 1 (rag_query.py: generate/SYSTEM, local or anthropic via
the RAG_BACKEND env var) — the handoff is text, so nothing is re-embedded.

Simplification for this educational step: we traverse only the TOP predicate.
The classifier returns a set (for questions spanning two relationships), but the
cutoff is still loose enough that the set is often "all five"; following the set
waits on a tighter cutoff. See doc/project_2_KG_RAG.md.
"""

import sys

import duckdb
from project_2_anchor import anchor
from project_2_disambiguate import RECALL_K, disambiguate, picked_predicates
from project_2_predicate_classifier import (  # noqa: F401 (classify used via picked_predicates)
    classify,
    load_config,
)
from project_2_traverse import traverse
from rag_query import BACKEND, SYSTEM, generate
from sentence_transformers import SentenceTransformer

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

MAX_FACTS = 40  # cap facts sent to the generator; truncation is reported, not silent


def name_of(text):
    """A node's display name: the part before the bracketed synonym list."""
    return text.split("[")[0].strip().rstrip(".")


def build_context(anchor_node, predicate, rows):
    """Turn traversed edges into readable triples, oriented by direction.

    rows are (other_id, category, other_text, dir) from traverse(); 'out' means
    the anchor is the subject, 'in' means it is the object.
    """
    anchor_name = name_of(anchor_node[2])
    pred = predicate.replace("biolink:", "")
    lines = []
    for _oid, _cat, other_text, direction in rows:
        other = name_of(other_text)
        if direction == "out":
            lines.append(f"{anchor_name} — {pred} — {other}")
        else:
            lines.append(f"{other} — {pred} — {anchor_name}")
    return anchor_name, "\n".join(lines)


def main():
    question = " ".join(sys.argv[1:]) or "What genes cause Ehlers-Danlos syndrome?"

    con = duckdb.connect(DB, read_only=True)
    embed_model, cutoff, predicates = load_config()
    model = SentenceTransformer(embed_model)
    pred_ids = list(predicates)
    pred_emb = model.encode(list(predicates.values()), convert_to_tensor=True)

    preds = picked_predicates(question, model, pred_ids, pred_emb, cutoff)
    if not preds:
        print("No predicate above cutoff — cannot choose what to traverse.")
        return

    candidates = anchor(con, model, question, k=RECALL_K)
    _rank, anchor_node, _count = disambiguate(con, candidates, preds)
    if anchor_node is None:
        print("No usable anchor found among the candidates.")
        return

    top_pred = preds[0]
    rows = traverse(con, anchor_node[0], top_pred)
    truncated = len(rows) > MAX_FACTS
    rows = rows[:MAX_FACTS]

    anchor_name, context = build_context(anchor_node, top_pred, rows)
    user = f"Facts about {anchor_name}:\n{context}\n\nQuestion: {question}"
    answer = generate(SYSTEM, user)

    print(f"Q: {question}   [backend: {BACKEND}]\n")
    print(f"Anchor:    {anchor_node[0]}  {anchor_name}")
    print(f"Predicate: {top_pred.replace('biolink:', '')}")
    n = len(rows)
    note = f" (capped from more; MAX_FACTS={MAX_FACTS})" if truncated else ""
    print(f"Facts:     {n} triples{note}")
    for line in context.splitlines()[:8]:
        print(f"  {line}")
    if n > 8:
        print(f"  ... and {n - 8} more")
    print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
