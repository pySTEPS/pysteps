.. _pystepsrc:

The pySTEPS configuration file (pystepsrc)
==========================================

.. _JSON: https://en.wikipedia.org/wiki/JSON
.. _AttrDict: https://pypi.org/project/attrdict/

The pysteps package allows the users to customize the default settings
and configuration.
The configuration parameters used by default are loaded from a user-defined
JSON_ file and then stored in the **pysteps.rcparams** AttrDict_.

The configuration parameters can be accessed as attributes or as items
in a dictionary. For e.g., to retrieve the default parameters
the following ways are equivalent::

    import pysteps

    # Retrieve the colorscale for plots
    colorscale = pysteps.rcparams['plot']['colorscale']
    colorscale = pysteps.rcparams.plot.colorscale

    # Retrieve the the root directory of the fmi data
    pysteps.rcparams['data_sources']['fmi']['root_path']
    pysteps.rcparams.data_sources.fmi.root_path

A less wordy alternative::

    from pysteps import rcparams
    colorscale = rcparams['plot']['colorscale']
    colorscale = rcparams.plot.colorscale

    fmi_root_path = rcparams['data_sources']['fmi']['root_path']
    fmi_root_path = rcparams.data_sources.fmi.root_path



When the pysteps package imported, it looks for **pystepsrc** file in the
following order:

- **$PWD/pystepsrc** : Looks for the file in the current directory
- **$PYSTEPSRC** : If the system variable $PYSTEPSRC is defined and it
  points to a file, it is used.
- **$PYSTEPSRC/pystepsrc** : If $PYSTEPSRC points to a directory, it looks for the
  pystepsrc file inside that directory.
- **$HOME/.pysteps/pystepsrc** (unix and Mac OS X) : If the system variable $HOME is defined, it looks
  for the configuration file in this path.
- **$USERPROFILE/pysteps/pystepsrc** (windows only): It looks for the configuration file
  in the pysteps directory located user's home directory.
- Lastly, it looks inside the library in pysteps/pystepsrc for a
  system-defined copy.

The recommended method to set-up the configuration files is to edit a copy
of the default **pystepsrc** file that is distributed with the package
and place that copy inside the user home folder.
See the instructions below.


Setting up the user-defined configuration file
----------------------------------------------


Linux and OSX users
~~~~~~~~~~~~~~~~~~~

For Linux and OSX users, the recommended way to customize the pysteps
configuration is place the pystepsrc parameters file in the users home folder
${HOME} in the following path: **${HOME}/.pysteps/pystepsrc**

To steps to setup up the configuration file in the home directory first we
need to create the directory if it does not exist. In a terminal, run::

    $ mkdir -p ${HOME}/.pysteps

The next step is to find the location of the library's pystepsrc file being
actually used.
When we import pysteps in a python interpreter, the configuration file loaded
is shown::

    import pysteps
    "Pysteps configuration file found at: /path/to/pysteps/library/pystepsrc"

Then we copy the library's default rc file to that directory::

    $ cp /path/to/pysteps/library/pystepsrc ${HOME}/.pysteps/pystepsrc

Edit the file with the text editor of your preference and change the default
configurations with your preferences.

Finally, check that the new updated file is being loaded by the library::

     import pysteps
     "Pysteps configuration file found at: /home/user_name/.pysteps/pystepsrc"


Windows
~~~~~~~

For windows users, the recommended way to customize the pySTEPS
configuration is place the pystepsrc parameters file in the users folder
(defined in the %USERPROFILE% environment variable) in the following path:
**%USERPROFILE%/pysteps/pystepsrc**

To steps to setup up the configuration file in the home directory first we
need to create the directory if it does not exist. In a terminal, run::

    $ mkdir -p %USERPROFILE%\pysteps

The next step is to find the location of the library's pystepsrc file being
actually used.
When we import pysteps in a python interpreter, the configuration file loaded
is shown::

    import pysteps
    "Pysteps configuration file found at: C:\path\to\pysteps\library\pystepsrc"

Then we copy the library's default rc file to that directory::

    $ cp C:\path\to\pysteps\library\pystepsrc %USERPROFILE%\pysteps\pystepsrc

Edit the file with the text editor of your preference and change the default
configurations with your preferences.

Finally, check that the new updated file is being loaded by the library::

     import pysteps
     "Pysteps configuration file found at: C:\User\Profile\.pysteps\pystepsrc"



More
----

.. toctree::
    :maxdepth: 1

    Example pystepsrc file <pystepsrc_example>
