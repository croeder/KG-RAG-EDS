"""Shared fixtures. Puts bin/ on the import path and loads the pipeline once.

These are integration fixtures: they use the real DuckDB subgraph and the real
all-MiniLM embedder, because the faults under test live in how embeddings rank
against real graph structure — a mock would test nothing.
"""

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bin"))


@pytest.fixture(scope="session")
def con():
    import duckdb

    return duckdb.connect(str(REPO / "data" / "eds.duckdb"), read_only=True)


@pytest.fixture(scope="session")
def pipeline():
    """The loaded classifier inputs: model, predicate ids, their embeddings, cutoff."""
    from project_2_predicate_classifier import load_config
    from sentence_transformers import SentenceTransformer

    embed_model, cutoff, predicates = load_config()
    model = SentenceTransformer(embed_model)
    pred_ids = list(predicates)
    pred_emb = model.encode(list(predicates.values()), convert_to_tensor=True)
    return {"model": model, "pred_ids": pred_ids, "pred_emb": pred_emb, "cutoff": cutoff}


@pytest.fixture(scope="session")
def gold():
    """Resolved gold cases keyed by id (eval/gold_eds.resolved.jsonl)."""
    lines = (REPO / "eval" / "gold_eds.resolved.jsonl").read_text().splitlines()
    return {json.loads(x)["id"]: json.loads(x) for x in lines if x.strip()}
