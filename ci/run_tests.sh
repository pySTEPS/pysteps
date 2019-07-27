#!/bin/bash

# Load pysteps-data
# -----------------
export PYSTEPS_BUILD_DIR="$( pwd )"
git clone https://github.com/pySTEPS/pysteps-data.git $PYSTEPS_BUILD_DIR/pysteps-data
python ci/create_pystepsrc_file.py
export PYSTEPSRC=$PYSTEPS_BUILD_DIR/pystepsrc.travis

# Run tests
cd ~
pytest --pyargs pysteps --cov=pysteps -ra;