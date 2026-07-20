"""Predicate classifier: known limitation + the guard a fix must not break.

The xfail below documents a repeatable limitation of the simple pipeline, not a
regression. strict=True means if it ever starts passing, pytest fails — a nudge to
close the issue and delete the marker.
"""

import pytest
from project_2_predicate_classifier import classify


def top_predicate(pipeline, question):
    ranked = classify(question, pipeline["model"], pipeline["pred_ids"], pipeline["pred_emb"])
    return ranked[0][0]


@pytest.mark.xfail(
    strict=True,
    reason="known limitation, see issue #1: 'type' in a subtype name routes phenotype questions to subclass_of",
)
def test_classify_WHEN_phenotype_question_names_a_subtype_by_type_word_SHOULD_rank_has_phenotype_top(pipeline):
    # "hypermobility type EDS": 'type' is part of the subtype NAME; the question
    # asks for phenotypes. The subclass_of description ("types, subtypes, or forms")
    # out-scores has_phenotype, so the top predicate is wrong.
    assert top_predicate(pipeline, "What are the phenotypes of hypermobility type EDS?") == (
        "biolink:has_phenotype"
    )


def test_classify_WHEN_question_asks_for_the_list_of_subtypes_SHOULD_rank_subclass_of_top(pipeline):
    # The guard: the legitimate subtype-listing question must keep routing to
    # subclass_of. Any fix for the case above must not break this.
    assert top_predicate(pipeline, "What are the types of Ehlers-Danlos syndrome?") == (
        "biolink:subclass_of"
    )
