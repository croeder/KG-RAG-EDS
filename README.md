# KG-RAG-EDS

Learning project: build a KG-RAG pipeline against a real biomedical knowledge graph, scoped
narrowly enough to finish. Distinct from the reasoning-methods comparison project (semantic
similarity vs. semantic embedding vs. network embedding on the full Monarch Graph) — that one
is shelved and aimed at a publishable result; this one is pedagogical, project-driven, and
deliberately small.

## Purpose

Learn KG-RAG mechanics hands-on rather than by reading straight through the reference books.
Reading is pulled in just-in-time against whatever the current build stage needs, not done
upfront.

## Scope

- **Data source:** [Monarch Initiative](https://monarchinitiative.org) knowledge graph, a
  subgraph pulled locally in KGX format, scoped to the Ehlers-Danlos syndrome (EDS)
  neighborhood — genes, phenotypes, related/differential diagnoses. Exact hop-radius and
  included edge/predicate types: **TBD**.
- **Access:** local pull, not live API queries against Monarch's hosted endpoint — for
  reproducibility and offline iteration. See `data/` below.
- **Out of scope:** the full Monarch graph, cross-species inference, anything that assumes the
  reasoning-methods comparison project's results.

## Sub-projects

Four sub-projects, not a sequence of steps. **Each one ends in a complete, working system
that answers questions about EDS.** They share infrastructure (the subgraph pull, embedding
utilities, an eval harness) and later ones build on earlier ones, but none is a gate: 2, 3,
and 4 are each a different working system rather than a stage on the way to a single one.
Building on each other doesn't stop each from being an individual deliverable. They are
individually sized pieces of work, addressed separately below.

### 1. Plain vector RAG

No graph structure. Text-embed KG node text (names, synonyms, descriptions) and retrieve by
nearest-neighbor search against the query embedding; feed the retrieved node text to the LLM.

- **Done when:** a typed EDS question returns an LLM answer grounded in the retrieved node text.
- **Depends on:** the subgraph pull and node-text extraction only.
- **Status:** done. `bin/embed_nodes.py` (embed → DuckDB), `bin/rag_query.py`
  (retrieve + local Qwen generation). See `doc/project_1_RAG.md`.

#### Local
Q: What joint problems are associated with EDS?

```
Retrieved (grounding):
  0.567  MONDO:0007842  joint laxity, familial. ['EDS 11 (formerly)', EDS XI, EDS11,
  0.492  MONDO:0007527  Ehlers-Danlos syndrome, periodontitis type. [EDS VIII, 'EDS 
  0.484  MONDO:0007525  Ehlers-Danlos syndrome, arthrochalasia type. [AEDS, EDS 7A, 
  0.479  MONDO:0010586  X-linked Ehlers-Danlos syndrome. [EDS V, EDS5, Ehlers-Danlos
  0.479  HP:0001373     Joint dislocation. [Joint dislocation, Joint dislocations]. 

A: The context describes several joint problems associated with Ehlers-Danlos syndrome:

1. **Generalized Joint Hypermobility**: Often complicated by dislocation of major joints, particularly the shoulder but in some cases the kneecap.

2. **Concurrent Periodontitis**: In the case of Ehlers-Danlos syndrome type 8 (EDS8), it can present with periodontitis.

3. **Arthrochalasia Type**: Characterized by severe joint hypermobility, congenital hip dislocation, fragile, hyperextensible skin, hypotonia, and kyphoscoliosis.

4. **X-Linked Ehlers-Danlos Syndrome**: Features include hyperextensible skin but mild joint hyperlaxity, common symptoms including severe joint hypermobility, congenital heart disease, hernias, and short stature.

These conditions highlight the diverse nature of joint involvement in different types of Ehlers-Danlos syndrome.
```

#### Anthropic
Q: What joint problems are associated with EDS?   [backend: anthropic]
```
Retrieved (grounding):
  0.567  MONDO:0007842  joint laxity, familial. ['EDS 11 (formerly)', EDS XI, EDS11,
  0.492  MONDO:0007527  Ehlers-Danlos syndrome, periodontitis type. [EDS VIII, 'EDS 
  0.484  MONDO:0007525  Ehlers-Danlos syndrome, arthrochalasia type. [AEDS, EDS 7A, 
  0.479  MONDO:0010586  X-linked Ehlers-Danlos syndrome. [EDS V, EDS5, Ehlers-Danlos
  0.479  HP:0001373     Joint dislocation. [Joint dislocation, Joint dislocations]. 

A: Based on the context, EDS is associated with several joint problems:

- **Generalized joint hypermobility/hyperlaxity** — a common feature across EDS types.
- **Joint dislocation** (displacement or malalignment of joints), particularly of major joints such as the shoulder and, in some cases, the kneecap (familial joint laxity type).
- **Congenital hip dislocation** — frequently reported in familial joint laxity and arthrochalasia types.
- **Joint instability**.

Additionally, the arthrochalasia type includes severe joint hypermobility along with related features like hypotonia and kyphoscoliosis.
```




### 2. KG-RAG — build on plain RAG by adding graph-native retrieval

Entity-link the question into the graph, then retrieve by structure — either literal edge
traversal or a text-to-SPARQL path — as a retrieval mechanism added alongside (1).

- **Done when:** a typed EDS question returns an answer grounded in facts pulled by walking or
  querying the graph.
- **Depends on:** (1).

### 3. Hybrid Text and KG RAG

Graph-retrieved facts and text-retrieved context feed the same generation step.

- **Done when:** a typed EDS question returns an answer grounded in both retrieval channels.
- **Depends on:** (1) and (2) — reuses the text channel from (1) and the graph channel from (2).

### 4. Apply / evaluate

Informal for now. Likely point where this repo starts sharing infrastructure (subgraph pull,
embedding utilities, eval harness) with the reasoning-methods comparison project once that's
unshelved.

- **Done when:** the working systems above can be exercised and compared on real EDS questions.
- **Depends on:** whichever of (1)–(3) is being evaluated.

Current stage: sub-project 1 (plain vector RAG) done; sub-project 2 (graph-native retrieval) next.

## Data

- Source: Monarch KG, [KGX format](https://github.com/biolink/kgx) (nodes/edges as TSV).
- **Source-of-truth convention:** the pulled dump is not hand-edited. Pull date and Monarch KG
  release version are pinned at pull time; updates happen by re-pulling, not patching.
- Location: `data/` — gitignored, not committed. Pull script documents how to regenerate.

## Tooling

- **Language:** Python (`kgx`, `monarch-py`) for the pull and everything downstream. Decided —
  the RAG/embedding/LLM stack (NLP with Transformers, HuggingFace) is Python-native throughout;
  R's `monarchr` is a nicer query interface but would mean splitting the pipeline across two
  ecosystems for no real benefit at this scale.
- **LLM assistance:** Claude Code used for infrastructure and scaffolding (data pull, glue code,
  eval harness). Core retrieval/traversal/generation logic is hand-written deliberately — the
  point of the project is building that intuition, not shipping fast.

## Reading (pulled in as needed, not read upfront)

- *Knowledge Graphs: Fundamentals, Techniques and Applications* — construction & embeddings
  chapters, ahead of stage 1–2.
- *Foundations of Semantic Web Technology* — RDF/OWL/SPARQL, ahead of stage 2 (esp. if using
  text-to-SPARQL).
- *Natural Language Processing with Transformers* — embeddings & semantic search chapters,
  primary reference for stage 1 and 3.
- *Understanding Deep Learning* (Prince) — reference only, pulled chapter-by-chapter (e.g.
  attention) when something in NLP w/ Transformers needs deeper grounding.

## Related

- Reasoning-methods comparison project (Monarch Graph, semantic similarity / semantic embedding
  / network embedding) — shelved, likely future infra-sharing point with this project.
- Lab notebook: [Substack — TBD name/link]. Narrative/reflective updates live there; this repo
  (README + issues) is the source of truth for actual status.

## License

Apache License 2.0 — see `LICENSE.md`.

## Open decisions

- [ ] Project/repo name
- [ ] Exact EDS subgraph scope (hop radius, entity/edge types to include)
- [x] Python vs. R — Python
- [ ] Whether this lives standalone or under an org (current lean: standalone repo, no org —
  see discussion; revisit only if a genuine fork/independent-deploy need shows up)
