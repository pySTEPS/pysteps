# -*- coding: utf-8 -*-
import os

from tempfile import NamedTemporaryFile

import pysteps
from pysteps import load_config_file

minimal_pystepsrc_file = """
// pysteps configuration
{
    "silent_import": false,
    "outputs": {
        "path_outputs": "./"
    },
    "plot": {
        "motion_plot": "quiver",
        "colorscale": "pysteps"
    },
    "data_sources": {
        "bom": {
            "root_path": "./radar/bom",
            "path_fmt": "prcp-cscn/2/%Y/%m/%d",
            "fn_pattern": "2_%Y%m%d_%H%M00.prcp-cscn",
            "fn_ext": "nc",
            "importer": "bom_rf3",
            "timestep": 6,
            "importer_kwargs": {
                "gzipped": true
            }
        }        
    }
}
"""


def test_read_paramsrc():
    """
    Test that the parameter file is read correctly and the resulting
    pysteps.paramsrc dict can be accessed by attributes too.
    """

    with NamedTemporaryFile(mode="w", delete=False) as tmp_paramsrc:
        tmp_paramsrc.write(minimal_pystepsrc_file)
        tmp_paramsrc.flush()

    # Perform a dry run that does not update
    # the internal pysteps.rcparams values.
    rcparams = load_config_file(tmp_paramsrc.name, dryrun=True, verbose=False)
    os.unlink(tmp_paramsrc.name)
    # Test item and attribute getters
    assert rcparams["data_sources"]["bom"]["fn_ext"] == "nc"
    assert rcparams.data_sources.bom.fn_ext == "nc"

    bom_datasource_as_dict = rcparams["data_sources"]["bom"]
    bom_datasource_as_attr = rcparams.data_sources.bom
    assert bom_datasource_as_dict is bom_datasource_as_attr
    bom_datasource = bom_datasource_as_attr

    timestep_as_dict = bom_datasource["timestep"]
    timestep_as_attr = bom_datasource.timestep
    assert timestep_as_dict == 6
    assert timestep_as_attr == 6
    assert timestep_as_dict is timestep_as_attr

    importer_kwargs_dict = bom_datasource["importer_kwargs"]
    importer_kwargs_attr = bom_datasource.importer_kwargs
    assert importer_kwargs_attr is importer_kwargs_dict

    assert importer_kwargs_attr["gzipped"] is importer_kwargs_attr.gzipped
    assert importer_kwargs_attr["gzipped"] is True

    # Test item and attribute setters
    rcparams.test = 4
    assert rcparams.test == 4
    assert rcparams.test is rcparams["test"]

    rcparams["test2"] = 4
    assert rcparams.test2 == 4
    assert rcparams.test2 is rcparams["test2"]

    rcparams.test = dict(a=1, b="test")
    assert rcparams.test == dict(a=1, b="test")
    assert rcparams.test["a"] == 1
    assert rcparams.test["b"] == "test"

    assert rcparams.test["a"] is rcparams["test"].a
    assert rcparams.test["b"] is rcparams["test"].b
