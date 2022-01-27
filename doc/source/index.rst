pysteps -- The nowcasting initiative
====================================

Pysteps is a community-driven initiative for developing and maintaining an easy
to use, modular, free and open source Python framework for short-term ensemble
prediction systems.

The focus is on probabilistic nowcasting of radar precipitation fields,
but pysteps is designed to allow a wider range of uses.

Pysteps is actively developed on GitHub__, while a more thorough description
of pysteps is available in the pysteps reference publication:

.. note::
   Pulkkinen, S., D. Nerini, A. Perez Hortal, C. Velasco-Forero, U. Germann,
   A. Seed, and L. Foresti, 2019:  Pysteps:  an open-source Python library for
   probabilistic precipitation nowcasting (v1.0). *Geosci. Model Dev.*, **12 (10)**,
   4185–4219, doi:`10.5194/gmd-12-4185-2019 <https://doi.org/10.5194/gmd-12-4185-2019>`_.

__ https://github.com/pySTEPS/pysteps

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: For users

   Installation <user_guide/install_pysteps>
   Gallery <../auto_examples/index>
   My first nowcast (Colab Notebook) <https://colab.research.google.com/github/pySTEPS/pysteps/blob/master/examples/my_first_nowcast.ipynb>
   API Reference <pysteps_reference/index>
   Example data <user_guide/example_data>
   Configuration file (pystepsrc) <user_guide/set_pystepsrc>
   Machine learning applications <user_guide/machine_learning_pysteps>
   Bibliography <zz_bibliography>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: For developers

    Contributing Guide <developer_guide/contributors_guidelines>
    Importer plugins <developer_guide/importer_plugins>
    Testing <developer_guide/test_pysteps>
    Building the docs <developer_guide/build_the_docs>
    Packaging <developer_guide/pypi>
    Publishing to conda-forge <developer_guide/update_conda_forge>
    GitHub repository <https://github.com/pySTEPS/pysteps>
