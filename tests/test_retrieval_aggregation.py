"""Project-2 retrieval: known limitation on umbrella-level questions.

The gold for an umbrella question aggregates facts across the disease's subtype
subtree. The simple pipeline anchors ONE node and traverses one hop, so it reaches
only that node's edges — recall on the aggregated gold is far below 1. This xfail
documents that; strict=True flips it to a failure if aggregation is ever added.
"""

import pytest
from eval_score import prf, project2_retrieval


@pytest.mark.xfail(
    strict=True,
    reason="known limitation, see issue #2: pipeline anchors one node, cannot aggregate across the subtype subtree",
)
def test_project2_retrieval_WHEN_umbrella_level_question_SHOULD_reach_facts_across_the_subtype_subtree(
    con, pipeline, gold
):
    case = gold["eds-genes-umbrella"]  # gold gene set aggregated over all EDS subtypes
    _anchor, _pred, rows = project2_retrieval(
        con, pipeline["model"], pipeline["pred_ids"], pipeline["pred_emb"], pipeline["cutoff"],
        case["question"],
    )
    recall = prf([r[0] for r in rows], case["answer_entities"])["r"]
    assert recall >= 0.8  # simple pipeline currently ~0.05
