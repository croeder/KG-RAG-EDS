"""Shared schema constants for the EDS raw TSV dumps.

The raw edge dump (data/eds_edges_raw.tsv) came out of grep, so it has no header
row and its column order is fixed by the source. That column list was being
copy-pasted across loaders; it lives here once so the scripts agree by import,
not by luck. Read-only source-of-truth layout — do not reorder to match a query.
"""

# Column order of eds_edges_raw.tsv (grep output, no header line).
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
# swallow following rows. all_varchar keeps everything as text (no type guessing).
READ_OPTS = "delim='\t', quote='', all_varchar=true"
