
# Project 1: Plain Vector RAG

## The problem and the approach

We want to answer questions about Ehlers-Danlos syndrome (EDS), and answer them more precisely than a language model's
general training would on its own. The aim is to ground each answer in a specific, chosen knowledge source rather than in
whatever the model happened to absorb during training — so the answer reflects that source and can be traced back to it.

The general approach is retrieval-augmented generation (RAG). We prepare the knowledge source for retrieval by dividing
it into pieces and embedding each piece into a vector with a transformer model. Embedding is what lets us search by
meaning (semantic search) instead of by literal words (lexical search). At query time we prepare the question the same
way — embed it into a vector — and compare it against the piece vectors to retrieve the closest ones. That retrieval step
fills the same role a SQL `WHERE ... LIKE` clause would, finding the relevant bits, while matching on meaning rather than
on characters. Finally we hand the retrieved pieces to an LLM, which turns them into a written answer.

In this project the knowledge source is a knowledge graph — the Monarch KG, scoped to EDS. The pieces come for free: each
graph node becomes one piece of text (its name, synonyms, and description), so there is no chunking to do. (With a plain
document corpus, dividing into pieces means chunking long text into passages so each is small enough to embed and to fit
in the prompt.) We use only the node text at this stage, not the graph's edges; using the graph structure for retrieval
is project 2.

 The shape of plain vector RAG. Four pieces: build a text corpus (one document per node), embed it into vectors, retrieve the nearest
  vectors to a query, then generate an answer from the retrieved text. All four are done — generation runs a local
  Qwen model (`Qwen/Qwen2.5-1.5B-Instruct`) via `transformers`, behind a swap-seam so the Anthropic API can replace it later.

  Embedding is one model used twice. The same embedding model runs on the corpus (once, up front) and on each query (at search time).
  That's not incidental — it's the whole mechanism. Because both go through the identical model, they land in the same geometric space,
  so "nearest vector" genuinely means "closest in meaning." Two different models would produce incomparable coordinates.

  Semantic vs. lexical. Your SQL-regex-toupper() instinct is lexical — it matches literal characters, so it only finds text containing
  the words you typed. Embeddings are semantic — they match meaning, so "loose joints that dislocate" finds "Joint subluxation" with
  zero shared words. The tradeoff: semantic search is fuzzier and less explainable (always returns a ranked something, no inspectable
  "why"), where lexical is exact and debuggable. Real systems often combine both; we did pure-embedding on purpose.

  The embedder and the LLM are cousins, not the same thing. Both are transformers, but different flavors: the embedder is an encoder
  (reads a string, returns one summary vector, doesn't generate); the LLM is a decoder (predicts tokens, so it can write). They're
  separate models with separate weights and — importantly — separate, incompatible vector spaces. What passes between the retrieval step
  and the generation step is plain text, never the embedder's vectors. The LLM re-reads the retrieved text from scratch with its own
  internal representations.

  Transformers vs. word2vec. Both are trained on text — that's not the difference. The difference is static vs. contextual: word2vec is
  a fixed lookup table (one permanent vector per word, context ignored), while a transformer computes a vector on the fly by reading the
  whole sentence, so the same word gets different vectors in different contexts. A sentence-transformer then pools those into one
  vector for the whole string.

  HuggingFace and model choice. The Hub is a hosting registry with hundreds of embedding models; sentence-transformers is the library
  that runs any of them. Models differ in size/speed, vector dimension, and what text they were trained on. We picked all-MiniLM-L6-v2 —
  small, general-purpose, 384-dimensional — the common starting default. Swapping models later is a one-line code change but forces
  re-embedding the entire corpus, because the old vectors are meaningless in the new model's space.

  Storage and search in DuckDB. Each node's vector lives in a fixed-size FLOAT[384] column — the type must match the model's dimension.
  Retrieval is a single SQL query: ORDER BY array_cosine_similarity(embedding, $query) DESC LIMIT k. array_cosine_similarity is built
  into DuckDB core (no extension needed at 832 rows — it just scans everything, instantly). Cosine similarity measures the angle between
  two vectors and ignores their length: 1.0 = same direction/meaning, 0 = unrelated. Length is thrown away on purpose, since you care
  about direction in meaning-space, not how long a node's text is.

  That's the conceptual spine. The one piece of the stage-1 "done when" still open is generation — handing the retrieved node texts to
  an LLM for a grounded answer.

## Concrete result

832 nodes embedded into `data/eds.duckdb`. A retrieval smoke test:

```
query: 'loose joints that dislocate easily'

0.687  HP:0005008     Large joint dislocations.
0.633  HP:0032153     Joint subluxation. A partial dislocation of a joint.
0.619  HP:0012095     Multiple joint dislocation.
0.596  HP:0001373     Joint dislocation.
0.590  HP:0006439     Radioulnar dislocation.
```

The words "loose" and "easily" appear in none of the retrieved node texts. It
matched on meaning — the semantic-vs-lexical point, on real data.

## Generation: local vs. Anthropic

The same question run through both generation backends. Retrieval is identical (same five nodes, same scores) — it does
not depend on which model writes the answer. Only the generated prose differs.

### Local (Qwen)

```
Q: What joint problems are associated with EDS?

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

### Anthropic (Claude)

```
Q: What joint problems are associated with EDS?   [backend: anthropic]

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

## Artifacts

- `bin/node_text.py` — builds the per-node text corpus (`data/eds_node_text.tsv`).
- `bin/embed_nodes.py` — embeds the corpus, writes the `nodes` table
  (`id`, `category`, `text`, `embedding FLOAT[384]`) into `data/eds.duckdb`.
- Model: `all-MiniLM-L6-v2`, 384-dim. DB: `data/eds.duckdb` (gitignored).

