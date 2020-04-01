#!/bin/bash

# Work from the pysteps build directory
cd $PYSTEPS_BUILD_DIR

# Load pysteps-data
# -----------------
echo "Download the pysteps example data"
export PYSTEPS_DATA_DIR=$PYSTEPS_BUILD_DIR/pysteps_data
python $PYSTEPS_BUILD_DIR/ci/fetch_pysteps_data.py

# Replace the default version with the modified.
# pysteps will load this the config file ($PWD/pystepsrc)
cp $PYSTEPS_DATA_DIR/pystepsrc ~/pystepsrc

# Run tests
echo "Run test suite"
cd ~
pytest --pyargs pysteps --cov=pysteps -ra;