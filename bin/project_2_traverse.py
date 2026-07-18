#!/usr/bin/env python3
"""One-hop graph traversal from an anchor node (README build seq #2).

Given an anchor node id and a predicate, collect the facts one edge away. This
is the graph-native retrieval step: no embeddings, just SQL over the edges table.

Direction is handled the general way (option 2 in doc/project_2_KG_RAG.md): the
anchor may be the subject OR the object of its edges — Disease is the subject of
has_phenotype but the object of gene_associated_with_condition — so we match
edges where the anchor is on either side and return whichever endpoint is *not*
the anchor, joined back to its node text. No per-predicate direction table.

Trade-off this accepts: for the symmetric subclass_of (Disease -> Disease) the
anchor legitimately sits on both sides, so this returns both subtypes and
superclasses. `dir` ('out' = anchor is subject, 'in' = anchor is object) is
reported so that over-return is at least visible.

Run: project_2_traverse.py <anchor_id> <predicate>, or no args for the test set.
"""

import sys

import duckdb

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

# (anchor_id, predicate) smoke tests: subject-side, object-side, and the
# symmetric subclass_of that over-returns.
TEST_CASES = [
    ("MONDO:0014139", "biolink:has_phenotype"),  # disease -> phenotypes (out)
    ("MONDO:0007522", "biolink:gene_associated_with_condition"),  # gene -> disease (in)
    ("MONDO:0007522", "biolink:subclass_of"),  # disease <-> disease (both)
]


def traverse(con, anchor_id, predicate):
    """Return facts one hop away as (other_id, category, text, dir).

    Matches edges with the anchor on either side; returns the *other* endpoint.
    dir is 'out' when the anchor is the subject, 'in' when it is the object.
    """
    return con.execute(
        """
        SELECT
          CASE WHEN e.subject = $anchor THEN e.object ELSE e.subject END AS other_id,
          n.category,
          n.text,
          CASE WHEN e.subject = $anchor THEN 'out' ELSE 'in' END AS dir
        FROM edges e
        JOIN nodes n
          ON n.id = CASE WHEN e.subject = $anchor THEN e.object ELSE e.subject END
        WHERE e.predicate = $pred
          AND (e.subject = $anchor OR e.object = $anchor)
        """,
        {"anchor": anchor_id, "pred": predicate},
    ).fetchall()


def main():
    con = duckdb.connect(DB, read_only=True)

    if len(sys.argv) > 2:
        cases = [(sys.argv[1], sys.argv[2])]
    else:
        cases = TEST_CASES

    for anchor_id, predicate in cases:
        rows = traverse(con, anchor_id, predicate)
        print(f"\n{anchor_id}  --{predicate.replace('biolink:', '')}-->  ({len(rows)} facts)")
        for other_id, category, text, direction in rows[:8]:
            cat = category.replace("biolink:", "")
            print(f"  [{direction}] {cat:18} {other_id:14} {text[:50]}")
        if len(rows) > 8:
            print(f"  ... and {len(rows) - 8} more")


if __name__ == "__main__":
    main()
