#!/bin/bash

# Work from the pysteps build directory
cd $PYSTEPS_BUILD_DIR

# Load pysteps-data
# -----------------
echo "Download pysteps example data"
git clone https://github.com/pySTEPS/pysteps-data.git $PYSTEPS_BUILD_DIR/pysteps-data
python $PYSTEPS_BUILD_DIR/ci/create_pystepsrc_file.py
export PYSTEPSRC=$PYSTEPS_BUILD_DIR/pystepsrc.travis

# Run tests
echo "Run test suite"
cd ~
pytest --pyargs pysteps --cov=pysteps -ra;