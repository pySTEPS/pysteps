#!/usr/bin/env bash

# exit on errors, and make sure to show it all
set -e -o pipefail

function install_correct_gdal_version() {
  IFS=" "
  version_string=$(dpkg -s libgdal-dev | grep -i version)
  read -ra version_array <<< "$version_string"
  IFS="+"
  read -ra version_value <<< "${version_array[1]}"
  version="${version_value[0]}"
  echo "Adding GDAL version ${version} to uv."
  uv add --optional gdal gdal[numpy]=="$version" --frozen
}

install_correct_gdal_version

