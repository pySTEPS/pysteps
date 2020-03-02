#!/bin/bash

# Work from the pysteps build directory
cd $PYSTEPS_BUILD_DIR

# Load pysteps-data
# -----------------
echo "Download the pysteps example data"
export PYSTEPS_DATA_DIR=$PYSTEPS_BUILD_DIR/pysteps_data
python $PYSTEPS_BUILD_DIR/ci/fetch_pysteps_data.py
export PYSTEPSRC=$PYSTEPS_BUILD_DIR/pystepsrc

# Run tests
echo "Run test suite"
cd ~
pytest --pyargs pysteps --cov=pysteps -ra;