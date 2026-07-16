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

## Build sequence

1. **Plain vector RAG** — no graph structure yet. Text-embed KG node text (names, synonyms,
   descriptions) and retrieve by nearest-neighbor search against the query embedding.
2. **Add graph-native retrieval** — entity linking into the graph, then either literal edge
   traversal or a text-to-SPARQL path, as a separate retrieval mechanism from (1).
3. **Combine into hybrid KG-RAG** — graph-retrieved facts and text-retrieved context feed the
   same generation step.
4. **Apply / evaluate** — informal for now. Likely point where this project starts sharing
   infrastructure (subgraph pull, embedding utilities, eval harness) with the reasoning-methods
   comparison project once that's unshelved.

Current stage: not yet started.

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
