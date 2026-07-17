#!/usr/bin/env python3
"""Embed each node's text and store the vectors in DuckDB (README build seq #1).

Reads the per-node text corpus, runs every row through one sentence-embedding
model, and writes a `nodes` table whose `embedding` column is a fixed-size
FLOAT[384] array. Retrieval is then a single ORDER BY array_cosine_similarity
query against this table.
"""

import duckdb
from sentence_transformers import SentenceTransformer

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
NODE_TEXT = f"{PROJECT_HOME}/data/eds_node_text.tsv"
DB = f"{PROJECT_HOME}/data/eds.duckdb"

# all-MiniLM-L6-v2: small, general-purpose, 384-dim. The dimension is fixed by
# the model; swapping models means re-embedding every row and changing DIM.
MODEL = "all-MiniLM-L6-v2"
DIM = 384

con = duckdb.connect(DB)

# eds_node_text.tsv was written by DuckDB COPY, so it is a clean quoted TSV:
# read it back with the default quoting (not the quote='' the raw pull needed).
rows = con.execute(
    f"SELECT id, category, text FROM read_csv('{NODE_TEXT}', delim='\t', header=true)"
).fetchall()

ids = [r[0] for r in rows]
cats = [r[1] for r in rows]
texts = [r[2] for r in rows]

# First call downloads the model weights to the HF cache, then runs locally.
model = SentenceTransformer(MODEL)
embeddings = model.encode(texts, show_progress_bar=True)

con.execute(f"""
    CREATE OR REPLACE TABLE nodes (
        id VARCHAR,
        category VARCHAR,
        text VARCHAR,
        embedding FLOAT[{DIM}]
    )
""")

con.executemany(
    "INSERT INTO nodes VALUES (?, ?, ?, ?)",
    [(ids[i], cats[i], texts[i], embeddings[i].tolist()) for i in range(len(ids))],
)

n = con.execute("SELECT count(*) FROM nodes").fetchone()[0]
print(f"{n} nodes embedded with {MODEL} ({DIM}-dim) -> {DB}")
con.close()
