.. _pysteps.io:

Input/output routines (:mod:`pysteps.io`)
*****************************************

Methods for browsing data archives, reading 2d precipitation fields and writing 
forecasts into files.

pysteps\.io\.interface
----------------------

.. automodule:: pysteps.io.interface
    :members:

pysteps\.io\.archive
--------------------

.. automodule:: pysteps.io.archive
    :members:

pysteps\.io\.importers
----------------------

.. currentmodule:: pysteps.io.importers

.. autosummary::
    import_bom_rf3
    import_fmi_pgm
    import_mch_gif
    import_mch_hdf5
    import_mch_metranet
    import_odim_hdf5

.. automodule:: pysteps.io.importers
    :members:

pysteps\.io\.readers
--------------------

.. automodule:: pysteps.io.readers
    :members:

pysteps\.io\.exporters
----------------------

.. currentmodule:: pysteps.io.exporters

.. autosummary::
    initialize_forecast_exporter_netcdf
    export_forecast_dataset
    close_forecast_file

.. automodule:: pysteps.io.exporters
    :members:

pysteps\.io\.nowcast\_importers
-------------------------------

.. automodule:: pysteps.io.nowcast_importers
    :members:
