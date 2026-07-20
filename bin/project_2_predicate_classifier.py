#!/usr/bin/env python3
"""Identify which predicate(s) a question is asking about (README build seq #2).

Approach 2 from doc/project_2_KG_RAG.md: embed a description of each predicate
with the same all-MiniLM model used in stage 1, embed the question, and rank
predicates by cosine similarity. Keep those at/above a cutoff (a set, not a
single top pick). No LLM in this path.

The tuning knobs — the predicate descriptions, the similarity cutoff, and the
embedding-model name — live in config/project_2_predicates.yaml, not in this
file, so they can be edited and diffed as data. See that file for why the
descriptions are hand-written rather than pulled from the Biolink Model.

Run with a question as args, or with no args to see the calibration spread over
built-in test questions.
"""

import pathlib
import sys

import yaml
from sentence_transformers import SentenceTransformer, util

PROJECT_HOME = str(pathlib.Path(__file__).resolve().parents[1])
CONFIG = f"{PROJECT_HOME}/config/project_2_predicates.yaml"

TEST_QUESTIONS = [
    "What are the symptoms of EDS?",
    "Which genes are associated with EDS?",
    "How is EDS inherited?",
    "What are the types of EDS?",
    "What genes and symptoms are linked to EDS?",
    "Tell me about EDS.",
]


def load_config(path=CONFIG):
    """Return (embed_model, cutoff, predicates dict) from the YAML config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg["embed_model"], cfg["cutoff"], cfg["predicates"]


def classify(question, model, pred_ids, pred_emb):
    """Return [(predicate, score)] sorted by descending cosine similarity."""
    q = model.encode(question, convert_to_tensor=True)
    sims = util.cos_sim(q, pred_emb)[0]
    return sorted(zip(pred_ids, sims.tolist()), key=lambda r: r[1], reverse=True)


def main():
    embed_model, cutoff, predicates = load_config()
    model = SentenceTransformer(embed_model)
    pred_ids = list(predicates)
    pred_emb = model.encode(list(predicates.values()), convert_to_tensor=True)

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else TEST_QUESTIONS

    for question in questions:
        print(f"\nQ: {question}")
        for pred, score in classify(question, model, pred_ids, pred_emb):
            mark = "*" if score >= cutoff else " "
            print(f"  {mark} {score:.3f}  {pred}")
    print(f"\n(* = at/above cutoff {cutoff})")


if __name__ == "__main__":
    main()
