#!/usr/bin/env bash


PROJECT_HOME=/Users/croeder/git/KG-RAG-EDS
cd $PROJECT_HOME

awk -F'\t'  '{if (!$21 && $1 ~ /^MONDO/ && $3 ~ /Ehlers-Danlos syndrome/)   print $1",  ",$3}' data/monarch-kg_nodes.tsv > data/eds_nodes.tsv

cut -d, -f1 data/eds_nodes.tsv > data/seeds.txt   

grep -F -f data/seeds.txt data/monarch-kg_edges.tsv > data/eds_edges_raw.tsv


# show categories
cut -f2,3 data/eds_edges_raw.tsv | sort | uniq -c | sort -rn
