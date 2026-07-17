#!/usr/bin/env python3
"""Fetch official Biolink Model descriptions for our predicates (build seq #2).

Approach 2 (doc/project_2_KG_RAG.md) matches a question against a description of
each predicate. Rather than hand-write those descriptions, pull the authoritative
`description` field from the Biolink Model schema YAML. Prints CURIE + slot name
+ description for our five predicates; those become the classifier's inputs.

Reproducible: the source URL and schema structure are explicit. PyYAML only (no
bmt/linkml runtime dependency).
"""

import sys
import urllib.request

import yaml

# Candidate raw locations for the schema (repo layout has changed over time);
# first one that loads and has a `slots` block wins.
SOURCES = [
    "https://raw.githubusercontent.com/biolink/biolink-model/master/biolink-model.yaml",
    "https://raw.githubusercontent.com/biolink/biolink-model/master/src/biolink_model/schema/biolink_model.yaml",
]

# Our predicates. Biolink slot names use spaces, not the CURIE underscores.
PREDICATES = {
    "biolink:has_phenotype": "has phenotype",
    "biolink:gene_associated_with_condition": "gene associated with condition",
    "biolink:causes": "causes",
    "biolink:has_mode_of_inheritance": "has mode of inheritance",
    "biolink:subclass_of": "subclass of",
}


def load_schema():
    for url in SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = yaml.safe_load(r.read())
        except Exception as e:  # noqa: BLE001 - report and try next
            print(f"  (skip {url}: {e})", file=sys.stderr)
            continue
        if isinstance(data, dict) and "slots" in data:
            return url, data
    raise SystemExit("no Biolink schema source loaded")


def main():
    url, data = load_schema()
    slots = data["slots"]
    print(f"# source: {url}\n")
    for curie, slot in PREDICATES.items():
        desc = (slots.get(slot) or {}).get("description", "<missing>")
        print(f"{curie}  ({slot})")
        print(f"    {desc}\n")


if __name__ == "__main__":
    main()
