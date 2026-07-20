#!/usr/bin/env python3
"""Derive gold answer sets for the eval cases (README build seq #4, project 4).

Reads the hand-authored cases in eval/gold_eds.yaml and, for each, runs an
INDEPENDENT oracle query over the edges table to produce answer_entities. Writes
eval/gold_eds.resolved.jsonl — the file the scorer consumes.

Independence is the point: the oracle here is plain SQL over edges, deliberately
NOT bin/project_2_traverse.py. Scoring the pipeline against a copy of the pipeline
would measure nothing. See eval/README.md and doc/project_4_evaluation.md.

The oracle:
  answer = { non-anchor endpoint of every <predicate> edge, in the given
             direction, incident to any node in the anchor set }
where the anchor set is {anchor} plus, when expand_subclasses is true, the
transitive subclass_of descendants of the anchor.

Run: venv/bin/python3 bin/eval_derive_gold.py
"""

import json
import pathlib

import duckdb
import yaml

# Path-relative so this survives the repo move; the older bin/ scripts hardcode a
# stale absolute PROJECT_HOME and no longer run.
REPO = pathlib.Path(__file__).resolve().parents[1]
DB = REPO / "data" / "eds.duckdb"
GOLD_IN = REPO / "eval" / "gold_eds.yaml"
GOLD_OUT = REPO / "eval" / "gold_eds.resolved.jsonl"

SUBCLASS = "biolink:subclass_of"


def subclass_descendants(con, root):
    """Transitive-downward closure over subclass_of, including root itself."""
    seen = {root}
    frontier = [root]
    while frontier:
        nxt = []
        for node in frontier:
            rows = con.execute(
                "SELECT subject FROM edges WHERE predicate = ? AND object = ?",
                [SUBCLASS, node],
            ).fetchall()
            for (child,) in rows:
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        frontier = nxt
    return seen


def oracle(con, anchor, predicate, direction, expand_subclasses):
    """Return the sorted set of non-anchor endpoints answering the case.

    direction: 'out' = anchor is subject, 'in' = anchor is object, 'both' = either.
    """
    anchors = subclass_descendants(con, anchor) if expand_subclasses else {anchor}
    answer = set()
    for a in anchors:
        if direction in ("out", "both"):
            rows = con.execute(
                "SELECT object FROM edges WHERE predicate = ? AND subject = ?",
                [predicate, a],
            ).fetchall()
            answer.update(o for (o,) in rows)
        if direction in ("in", "both"):
            rows = con.execute(
                "SELECT subject FROM edges WHERE predicate = ? AND object = ?",
                [predicate, a],
            ).fetchall()
            answer.update(s for (s,) in rows)
    # never count an anchor node as its own answer
    answer -= anchors
    return sorted(answer)


def main():
    con = duckdb.connect(str(DB), read_only=True)
    spec = yaml.safe_load(GOLD_IN.read_text())
    cases = spec["cases"]

    resolved = []
    for c in cases:
        direction = c.get("direction", "both")
        entities = oracle(
            con,
            c["anchor"],
            c["predicate"],
            direction,
            c.get("expand_subclasses", False),
        )
        resolved.append(
            {
                "id": c["id"],
                "question": c["question"],
                "anchor": c["anchor"],
                "predicate": c["predicate"],
                "direction": direction,
                "expand_subclasses": c.get("expand_subclasses", False),
                "answer_entities": entities,
            }
        )

    with GOLD_OUT.open("w") as f:
        for r in resolved:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(resolved)} cases -> {GOLD_OUT.relative_to(REPO)}")
    for r in resolved:
        n = len(r["answer_entities"])
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"  {n:4d}  {r['id']}{flag}")


if __name__ == "__main__":
    main()
