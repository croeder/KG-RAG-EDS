# Gold standard — shared case format (v1 and v2)

The eval gold set for the KG-RAG line. This format is shared by **KG-RAG-EDS (v1)**
and **KG-RAG-Monarch (v2)**: same case shape, same scorer, different way of *filling*
the cases. It is the concrete build of the design reasoning in
`doc/project_4_evaluation.md` — read that first for *why*; this file is *what*.

## The core idea: the graph is its own answer key

A gold case never carries a hand-labeled answer. It carries a **question** and a
**query spec** — an assertion that "this question corresponds to this structural
query." The answer entities are then *derived* by running an independent oracle
(`bin/eval_derive_gold.py`) against the graph. The graph is both the retrieval
source and the answer key, so correctness/recall come almost for free; the only
human cost is authoring the question↔query correspondence.

Consequences of deriving rather than labeling:

- **Regenerable.** On a graph re-pull the answers are re-derived, never patched by
  hand — consistent with the source-of-truth convention (`README.md` → Data).
- **Honest oracle.** The oracle is a small, obviously-correct SQL over the edges
  table — deliberately *not* `bin/project_2_traverse.py`. Scoring the pipeline
  against a reimplementation of the pipeline would test nothing. The oracle and the
  system under test must be independent.

## Two files per project

- `gold_<project>.yaml` — **hand-authored**, the source of truth. Questions and
  query specs only. No answer entities.
- `gold_<project>.resolved.jsonl` — **generated** by the deriver. One case per line,
  now carrying `answer_entities`. This is what the scorer consumes. Not hand-edited.

## Case fields (authored)

```yaml
- id: eds-genes-umbrella        # stable, unique
  question: "What genes are associated with Ehlers-Danlos syndrome?"
  anchor: MONDO:0020066         # the node the question is about (entity-linking target)
  predicate: biolink:gene_associated_with_condition
  direction: in                 # out = anchor is subject | in = anchor is object | both
  expand_subclasses: true       # union the traversal over anchor + its subclass descendants
  notes: "umbrella has no direct gene edges; needs subtype aggregation"
```

Field meanings:

- **anchor** — the disease node the question links to. Authoring this is the
  "assert question↔query" step. For questions that name the whole disease, anchor is
  the umbrella (`MONDO:0020066`); for questions about one subtype, anchor is that
  subtype node.
- **predicate / direction** — which edges answer the question, and which side the
  anchor sits on. Direction matters because the anchor is subject for some predicates
  (`has_phenotype`: disease→phenotype, `out`) and object for others
  (`gene_associated_with_condition`: gene→disease, `in`). The oracle returns the
  *non-anchor* endpoint.
- **expand_subclasses** — the umbrella-vs-subtype knob made explicit. In this
  subgraph the umbrella `MONDO:0020066` carries **only** `subclass_of` edges;
  phenotypes, genes, and inheritance hang off the subtype nodes. So an umbrella-
  anchored question like "genes that cause EDS" is only answerable by unioning the
  traversal over the umbrella *and* its subclass descendants (42 nodes here). Setting
  this to `true` is asserting "the correct answer aggregates across subtypes." It is
  itself a thing the retrieval side can get wrong, so it belongs in the gold.

## Derived field (generated)

- **answer_entities** — list of CURIEs the oracle returned. The gold answer set.
  Retrieval scoring is precision/recall of the pipeline's pulled facts against this.

## How v1 and v2 differ

Same format; the difference is **how cases are produced**:

- **v1 (EDS)** — hand-authored, small, legible. One disease, a spread of questions
  across the five predicates the project-2 classifier can route to
  (`config/project_2_predicates.yaml`), deliberately exercising both umbrella-anchored
  (`expand_subclasses: true`) and subtype-anchored cases. This is `gold_eds.yaml`.
- **v2 (Monarch)** — mostly **synthetic**, because the point of v2 is scale and hand-
  authoring across the broad graph does not scale. Sample a seed node, sample an edge
  or subgraph, template a question from it; the gold answer is the sampled edge target.
  Stratify the sample by predicate and node degree so the disambiguation and degree
  heuristics get exercised. Keep a small hand-authored, realistically-phrased subset
  alongside it to cover synthetic phrasing's blind spot. v2 has no loaded graph yet, so
  its `gold_monarch.yaml` and generator are stubbed until the v2 subgraph exists.

## What this measures (and doesn't)

Per `doc/project_4_evaluation.md`, keep the two axes apart:

- **Retrieval quality** — precision/recall of pulled facts vs `answer_entities`. Uses
  this gold. Scores project 1 (did nearest-neighbor search surface those entities'
  node text in top-k?) and project 2 (did anchor→predicate→traverse reach them?).
- **Groundedness / faithfulness** — answer-vs-retrieved-context. **Needs no gold**, so
  it lives in the scorer, not here.

Correctness needs this reference; groundedness does not.
