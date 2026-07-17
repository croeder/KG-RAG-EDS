#!/usr/bin/env python3
"""Load the EDS subgraph into DuckDB and join edges to node names."""

import duckdb

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
NODES = f"{PROJECT_HOME}/data/monarch-kg_nodes.tsv"
EDGES = f"{PROJECT_HOME}/data/eds_edges_raw.tsv"

EDGE_COLS = [
    "id", "predicate", "category", "agent_type", "aggregator_knowledge_source",
    "knowledge_level", "primary_knowledge_source", "file_source", "provided_by",
    "publications", "qualifiers", "has_evidence", "object_specialization_qualifier",
    "original_predicate", "FDA_adverse_event_level", "disease_context_qualifier",
    "object_category", "subject_category", "frequency_qualifier", "has_count",
    "has_percentage", "has_quotient", "has_total", "negated", "onset_qualifier",
    "sex_qualifier", "has_attribute", "object_aspect_qualifier",
    "species_context_qualifier", "stage_qualifier", "qualifier", "subject",
    "object", "original_subject", "original_object",
]

# quote='' matters: node descriptions contain unbalanced quote characters that
# would otherwise swallow following rows.
READ_OPTS = "delim='\t', quote='', all_varchar=true"

con = duckdb.connect()

con.execute(f"""
    CREATE VIEW nodes AS
    SELECT * FROM read_csv('{NODES}', header=true, {READ_OPTS})
""")

con.execute(f"""
    CREATE VIEW edges AS
    SELECT * FROM read_csv('{EDGES}', header=false, names={EDGE_COLS}, {READ_OPTS})
    WHERE id <> 'id'
""")

print(con.execute("SELECT count(*) FROM edges").fetchone()[0], "edges")

OUT = f"{PROJECT_HOME}/data/eds_triples.tsv"
con.execute(f"""
    COPY (
        SELECT s.name AS subject, e.predicate, o.name AS object
        FROM edges e
        JOIN nodes s ON s.id = e.subject
        JOIN nodes o ON o.id = e.object
    ) TO '{OUT}' (HEADER, DELIMITER '\t')
""")
print("wrote", OUT)
