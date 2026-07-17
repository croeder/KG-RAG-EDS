#!/usr/bin/env python3
"""Explore the EDS subgraph edges (README build seq #2, orientation step).

Before building graph-native retrieval we need the traversal target to be
concrete: which predicates connect things, and which node categories they
connect. Prints edge counts by predicate and by subject->object category pair.
Read-only; touches nothing.
"""

import duckdb

PROJECT_HOME = "/Users/croeder/git/KG-RAG-EDS"
EDGES = f"{PROJECT_HOME}/data/eds_edges_raw.tsv"

# eds_edges_raw.tsv came out of grep, so it has no header on line 1.
EDGE_COLS = [
    "id",
    "predicate",
    "category",
    "agent_type",
    "aggregator_knowledge_source",
    "knowledge_level",
    "primary_knowledge_source",
    "file_source",
    "provided_by",
    "publications",
    "qualifiers",
    "has_evidence",
    "object_specialization_qualifier",
    "original_predicate",
    "FDA_adverse_event_level",
    "disease_context_qualifier",
    "object_category",
    "subject_category",
    "frequency_qualifier",
    "has_count",
    "has_percentage",
    "has_quotient",
    "has_total",
    "negated",
    "onset_qualifier",
    "sex_qualifier",
    "has_attribute",
    "object_aspect_qualifier",
    "species_context_qualifier",
    "stage_qualifier",
    "qualifier",
    "subject",
    "object",
    "original_subject",
    "original_object",
]

# quote='' matters: descriptions contain unbalanced quotes that would otherwise
# swallow following rows.
READ_OPTS = "delim='\t', quote='', all_varchar=true"

con = duckdb.connect()
con.execute(f"""
    CREATE VIEW edges AS
    SELECT * FROM read_csv('{EDGES}', header=false, names={EDGE_COLS}, {READ_OPTS})
    WHERE id <> 'id'
""")

total = con.execute("SELECT count(*) FROM edges").fetchone()[0]
print(f"total edges: {total}\n")

print("-- predicates --")
for pred, c in con.execute(
    "SELECT predicate, count(*) c FROM edges GROUP BY 1 ORDER BY c DESC"
).fetchall():
    print(f"{c:6}  {pred}")

print("\n-- subject_category -> object_category (top 15) --")
for subj, obj, c in con.execute(
    """SELECT subject_category, object_category, count(*) c
       FROM edges GROUP BY 1, 2 ORDER BY c DESC LIMIT 15"""
).fetchall():
    print(f"{c:6}  {subj}  ->  {obj}")
