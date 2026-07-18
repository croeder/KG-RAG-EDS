# Project 2: KG-RAG (graph-native retrieval)

## The problem and the approach

Same goal as project 1 — answer questions about Ehlers-Danlos syndrome (EDS), grounded in a specific knowledge source
rather than in a model's general training. What changes is *how we retrieve*. Project 1 retrieved by text similarity: it
embedded the question and found node text that meant something similar. Project 2 retrieves by **graph structure**: it
finds the facts the question is about by following the graph's edges (or by querying the graph), and uses the graph as a
graph. This is what makes it KG-RAG rather than plain RAG. It is added alongside project 1, not a replacement — the same
Monarch KG, scoped to EDS, and the same generation step at the end.

Graph-native retrieval has two parts that text retrieval did not need:

1. **Anchor (entity-linking).** Get from the words of the question to a specific node (or nodes) in the graph — e.g. from
   "What genes are associated with EDS?" to the EDS disease node. This is the genuinely new problem in project 2.
2. **Traverse.** From the anchor node, follow the edges the question calls for and collect the connected facts. Which
   edges to follow is set by what the question asks (see the predicate menu below). The retrieved facts are what ground
   the answer.

Once facts are retrieved, generation is unchanged from project 1: hand the pulled facts to an LLM, which writes the
answer. The retrieval channel is new; the generation channel is reused.

**Where the embeddings are — and aren't.** Embeddings appear at exactly two spots, both at the *entry*, where English has
to get onto the graph: resolving the question to an anchor node, and picking which predicate(s) to follow. Both are stage
1's trick reused — the question's words won't literally match a node's label or a predicate's name, so we compare by
meaning in vector space instead. Everything *after* that is plain SQL over the edges table: from the anchor node, follow
the chosen predicate to the nodes it connects to. The traverse is graph connectivity, not similarity — there are no
embeddings once you are on the graph. So the full pipeline is: question → *(embed)* anchor node; question → *(embed)*
predicate set; then *(SQL)* walk the edges → facts → the project-1 generation seam.

## What the graph can answer (the predicate menu)

The EDS slice has ~1900 edges. An edge's *type* — its predicate — determines what kind of fact it carries, so the set of
predicates is the menu of what the graph can answer, and the question determines which to follow:

| Question about… | Predicate to follow |
|---|---|
| symptoms / features of EDS | `has_phenotype` (1628 — the bulk) |
| genes involved | `gene_associated_with_condition`, `causes` |
| how it's inherited | `has_mode_of_inheritance` |
| EDS subtypes / hierarchy | `subclass_of` |

(`related_to`, `has_disease`, `model_of`, `in_taxon` are peripheral here — generic links, animal models, species.)

The design fork of project 2 is **who chooses the predicate and builds the query**, from least to most LLM involvement:

1. Code maps the question to a predicate and runs a fixed traversal — no LLM in retrieval.
2. The LLM picks the predicate (or a small plan); code runs the traversal.
3. The LLM writes the query itself (text-to-SQL).

Our graph lives in DuckDB, so the query mechanism is SQL over the edges table — not SPARQL, which would require moving the
data into an RDF triplestore. (Inventory produced by `bin/project_2_explore_edges.py`.)

## Identifying the predicate

Choosing which predicate to follow is a classifier: input is the question, output is one or more predicates from the menu
above. Three ways to build it, in increasing robustness and cost:

1. **Keyword rules.** A hand-written map from trigger words to predicates ("symptom" → `has_phenotype`, "gene" →
   `gene_associated_with_condition` / `causes`, and so on). Transparent and dependency-free, but lexical — brittle to
   phrasing and synonyms, the same fragility stage 1 escaped.
2. **Embedding match.** Give each predicate a one-line description, embed each predicate in the menu with the same
   `all-MiniLM` model used in stage 1, embed the question, and take the nearest by cosine similarity. The hand-written
   descriptions in the config supply the *words*; the embedding does the *matching* — same config text as approach 1, but
   compared by meaning instead of by keyword, so "which mutations cause it" can reach the gene predicate without sharing a
   word with it. This is stage 1's semantic search applied to the predicate menu instead of node text. Reuses the existing
   embedder, tolerant of phrasing, and keeps retrieval LLM-free. Failure mode: a vague question landing between two
   predicates.
3. **LLM classification.** One call — give the LLM the predicates and the question, constrain the output to the menu
   (structured output). Most robust, and naturally returns a set for multi-predicate questions, at the cost of an LLM call
   in the retrieval path.

**Decision: approach 2 (embedding match)** — the direct extension of stage 1, semantic rather than lexical, no LLM in
retrieval. Escalate to (3) if it proves too coarse on real questions. The classifier returns a *set* of predicates (those
at or above a similarity cutoff — currently `0.30` in `config/project_2_predicates.yaml`, eyeballed rather than calibrated
against a labeled question set), not a single one, so a question spanning two relationships ("genes and symptoms") is not
forced to pick one.

**What the descriptions say matters more than that they're authoritative.** The first attempt fed each predicate the
official Biolink Model `description` (pulled by `bin/project_2_fetch_biolink_defs.py`). It calibrated badly: on a symptoms
question `has_phenotype` lost to `has_mode_of_inheritance`, and `subclass_of` never won a "types of EDS" question. Ontology
definitions are written in ontology vocabulary, which shares few words — and so little embedding neighborhood — with the
way people actually ask. Replacing them with hand-written, question-facing descriptions (the words that show up in the
questions themselves) put the right predicate on top for every calibration question. Those descriptions live in
`config/project_2_predicates.yaml`; because they're hand-tuned data, they belong in config, not buried in code.

## Anchoring the question, and disambiguating with the predicate

The anchor is embedding recall: embed the question with the stage-1 model and take the nearest nodes
(`bin/project_2_anchor.py`). But *nearest by meaning* is not the same as *right to anchor on*. Of the 832 nodes, 635 are
phenotypes and only 85 are diseases, so the single nearest node is often a symptom, not the disease. Observed: "how is the
stretchy-skin condition inherited?" returns five skin-phenotype nodes and no disease at all; "hypermobility disorder" puts
the phenotype *Joint hypermobility* above the EDS disease node. Traversing from a symptom is the wrong starting point.

The fix is graph-native and reuses the predicate we already picked. The first thing tried — "keep any candidate that has
an edge of the picked predicate" — does not work, and the failure is instructive: a symptom is the *object* of a
`has_phenotype` edge, so it passes an "has an edge" test and gets chosen. Matching on either side is right for the
*traversal* (you already know the anchor, you just grab the other end) but wrong for *choosing* the anchor, because the
entity-asked-about and the answer sit on opposite ends of the very same edge. The loose predicate cutoff made it worse —
with `has_phenotype` in nearly every question's predicate set, phenotypes qualify for almost anything.

The teachable fix is the simplest one that uses the graph: **among the candidates, pick the one with the *most* edges of
the picked predicate(s) — the hub.** The disease has 100+ `has_phenotype` edges; a leaf phenotype has a handful, so
counting swamps the leaves and the disease wins even when it ranked low by embedding. No `category = Disease` rule, no
per-predicate direction table (both were considered and dropped as more machinery than the lesson needs), and it holds
even with the loose cutoff. The predicate constrains the anchor; the graph's own connectivity does the disambiguation.

Two honest limits, kept in view:

- **Recall depth is a knob.** The disease must be *in* the candidate pool to be counted, and at k=5 it sometimes is not
  (the "stretchy-skin" question surfaced only phenotypes in its top five; the right disease was rank 15). So anchor recall
  runs with a generous k and the count then narrows.
- **It's a heuristic** — it assumes the anchor is well-connected (true here) — and it picks the *most-connected* disease,
  which can be a well-annotated **subtype** (e.g. classic-like 2, 129 phenotype edges) rather than the **umbrella** EDS
  node. So "symptoms of EDS" may return one subtype's symptoms, not all of EDS. Umbrella-vs-subtype is a later refinement.

## Traversing the edges (direction matters)

With an anchor node and a predicate chosen, the one-hop traversal looks like it should be a single fixed template —
`SELECT object FROM edges WHERE subject = <anchor> AND predicate = <picked>`. It is not, because the anchor is not always
the *subject* of its edges. Checking the actual EDS slice (join each edge's subject and object back to `nodes.category`):

| predicate | direction | the disease anchor is the… |
|---|---|---|
| `has_phenotype` | Disease → PhenotypicFeature | subject (collect objects) |
| `has_mode_of_inheritance` | Disease → PhenotypicFeature | subject (collect objects) |
| `subclass_of` | Disease → Disease | subject; points *up* to the parent |
| `causes` | Gene → Disease | **object** (collect subjects) |
| `gene_associated_with_condition` | Gene → Disease | **object** (collect subjects) |

So for symptom / inheritance / subtype questions the anchor is the subject and you gather objects; for the two gene
predicates the anchor is the *object* and you gather subjects. A subject-only template returns nothing for "what genes
cause EDS?". Two related findings from the same check:

- `causes` and `gene_associated_with_condition` are the same 22 Gene→Disease edges — the gene questions collapse to one
  traversal.
- `subclass_of` is direction-sensitive in a trap way: it is child→parent, so "what are the *types* of EDS?" wants the
  edges where EDS is the **object** (its subtypes point up at it). Traverse it as the subject and you get EDS's parents —
  the opposite of the question.

This forces one design choice — how the traversal knows which side the anchor sits on for a given predicate:

1. **Hand-encode direction per predicate** in the config, alongside the descriptions (`has_phenotype: anchor_is=subject`,
   `causes: anchor_is=object`, …). Precise, but more hand-tuned data — the thing we've been trying not to accumulate.
2. **Generic — anchor on either side, take the other endpoint:** `WHERE predicate = ? AND (subject = ? OR object = ?)`,
   return whichever endpoint is not the anchor. No per-predicate table, fully general. Cost: for the symmetric
   `subclass_of` (Disease → Disease) the anchor legitimately sits on both sides, so it over-returns — both subtypes and
   superclasses — exactly for the one predicate where direction carries meaning.

(Directionality produced by joining `data/eds_edges_raw.tsv` back to the `nodes` categories; edges are not yet loaded as a
table in `eds.duckdb`, only read as a view over the TSV.)

- **Done when:** a typed EDS question returns an answer grounded in facts pulled by walking or querying the graph.
- **Depends on:** project 1 (generation seam) and the edges already loaded by `bin/load_duckdb.py`.

## Artifacts

- `bin/project_2_explore_edges.py` — read-only edge inventory (predicate counts); produced the menu above.
- `bin/project_2_fetch_biolink_defs.py` — pulls the official Biolink Model descriptions; used to show why they matched
  questions poorly.
- `bin/project_2_predicate_classifier.py` — the approach-2 classifier: embeds the question and the predicate descriptions,
  ranks by cosine similarity, and returns the set at/above the cutoff. Run with no args for the calibration spread.
- `config/project_2_predicates.yaml` — the tuning knobs: embedding-model name, cutoff, and the hand-written predicate
  descriptions.
