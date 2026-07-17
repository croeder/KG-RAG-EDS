#!/usr/bin/env python3
"""Build one text document per node in the EDS subgraph (README build seq #1).

Nodes = every distinct entity appearing as subject or object in the filtered
edges. Text = name + synonyms + description concatenated per node.
"""

import duckdb

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
NODES = f"{PROJECT_HOME}/data/monarch-kg_nodes.tsv"
EDGES = f"{PROJECT_HOME}/data/eds_edges_raw.tsv"
OUT = f"{PROJECT_HOME}/data/eds_node_text.tsv"

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

# distinct entity IDs = union of both edge endpoints
con.execute("""
    CREATE VIEW node_ids AS
    SELECT subject AS id FROM edges
    UNION
    SELECT object AS id FROM edges
""")

# concat_ws skips NULLs, so a node missing description/synonym just contributes
# fewer fields rather than blank separators.
con.execute(f"""
    COPY (
        SELECT
            n.id,
            n.category,
            concat_ws('. ', n.name, n.synonym, n.description) AS text
        FROM node_ids i
        JOIN nodes n ON n.id = i.id
    ) TO '{OUT}' (HEADER, DELIMITER '\t')
""")

n = con.execute("SELECT count(*) FROM node_ids").fetchone()[0]
print(n, "distinct nodes ->", OUT)

# coverage: how many have a real description vs. name-only
cov = con.execute("""
    SELECT
        count(*) FILTER (WHERE n.description IS NOT NULL AND n.description <> '') AS with_desc,
        count(*) AS total
    FROM node_ids i JOIN nodes n ON n.id = i.id
""").fetchone()
print(f"{cov[0]}/{cov[1]} nodes have a non-empty description")
