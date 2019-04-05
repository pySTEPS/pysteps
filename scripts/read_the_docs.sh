#!/bin/bash

export PYSTEPS_BUILD_DIR="$( pwd )"
git clone https://github.com/pySTEPS/pysteps-data.git $PYSTEPS_BUILD_DIR/pysteps-data
python scripts/read_the_docs.py
export PYSTEPSRC=$PYSTEPS_BUILD_DIR/pystepsrc.rtd