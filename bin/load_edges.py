#!/usr/bin/env python3
"""Load the EDS edges into eds.duckdb as a table (README build seq #2).

Stage 1 (embed_nodes.py) created eds.duckdb with only the `nodes` table. Graph
traversal needs the edges in the same database so a one-hop walk can join an
edge's endpoints back to node text in a single query. This reads the raw edge
TSV and writes an `edges` table alongside `nodes`. Re-runnable: it drops and
recreates the table, so it can be re-run after a fresh pull.
"""

import pathlib

import duckdb
from eds_schema import EDGE_COLS, READ_OPTS

PROJECT_HOME = str(pathlib.Path(__file__).resolve().parents[1])
DB = f"{PROJECT_HOME}/data/eds.duckdb"
EDGES = f"{PROJECT_HOME}/data/eds_edges_raw.tsv"

con = duckdb.connect(DB)
con.execute("DROP TABLE IF EXISTS edges")
con.execute(f"""
    CREATE TABLE edges AS
    SELECT * FROM read_csv('{EDGES}', header=false, names={EDGE_COLS}, {READ_OPTS})
    WHERE id <> 'id'
""")
n = con.execute("SELECT count(*) FROM edges").fetchone()[0]
print(f"loaded {n} edges into {DB}")
