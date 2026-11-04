#!/usr/bin/env bash

# exit on errors, and make sure to show it all
set -e -o pipefail

function install_correct_gdal_version() {
  old_IFS=$IFS
  IFS=" "
  version_string=$(dpkg -s libgdal-dev | grep -i version)
  read -ra version_array <<< "$version_string"
  IFS="+"
  read -ra version_value <<< "${version_array[1]}"
  version="${version_value[0]}"
  IFS="$old_IFS"
  uv add --optional gdal gdal[numpy]=="$version"
}

install_correct_gdal_version

