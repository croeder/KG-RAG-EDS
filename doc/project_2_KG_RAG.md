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
   `all-MiniLM` model used in stage 1, embed the question, and take the nearest by cosine similarity. This is stage 1's semantic search applied to
   the predicate menu instead of node text. Reuses the existing embedder, tolerant of phrasing, and keeps retrieval
   LLM-free. Failure mode: a vague question landing between two predicates.
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
