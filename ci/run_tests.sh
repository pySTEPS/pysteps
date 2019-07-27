#!/bin/bash

# Load pysteps-data
# -----------------
echo "Download pysteps example data"
export PYSTEPS_BUILD_DIR="$( pwd )"
git clone https://github.com/pySTEPS/pysteps-data.git $PYSTEPS_BUILD_DIR/pysteps-data
python ci/create_pystepsrc_file.py
export PYSTEPSRC=$PYSTEPS_BUILD_DIR/pystepsrc.travis

# Run tests
echo "Run test suite"
cd ~
pytest --pyargs pysteps --cov=pysteps -ra;