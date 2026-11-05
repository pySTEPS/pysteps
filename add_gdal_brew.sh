#!/usr/bin/env bash

# exit on errors, and make sure to show it all
set -e -o pipefail

function install_correct_gdal_version() {
  version="$(brew info --json gdal | jq -r '(.[] | (.installed[] | .version))')"
  IFS="_"
  read -ra version_value <<< "${version[0]}"
  version="${version_value[0]}"
  echo "Adding GDAL version ${version} to uv."
  uv add --optional gdal gdal[numpy]=="$version" --frozen
}

install_correct_gdal_version
