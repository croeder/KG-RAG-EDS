
# Project 1: Plain Vector RAG

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

## Artifacts

- `bin/node_text.py` — builds the per-node text corpus (`data/eds_node_text.tsv`).
- `bin/embed_nodes.py` — embeds the corpus, writes the `nodes` table
  (`id`, `category`, `text`, `embedding FLOAT[384]`) into `data/eds.duckdb`.
- Model: `all-MiniLM-L6-v2`, 384-dim. DB: `data/eds.duckdb` (gitignored).

