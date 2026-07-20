#!/usr/bin/env bash

# pull a monarch release. 

PROJECT_HOME="$(cd "$(dirname "$0")/.." && pwd)"
ls $PROJECT_HOME/data
cd $PROJECT_HOME/data
#wget https://data.monarchinitiative.org/monarch-kg/latest/monarch-kg.tar.gz

pwd
ls
echo ""
#gunzip monarch-kg.tar.gz
tar xvzf monarch-kg.tar
