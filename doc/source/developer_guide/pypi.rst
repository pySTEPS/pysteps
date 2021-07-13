.. _pypi_relase:

=============================
Packaging the pysteps project
=============================

The `Python Package Index <https://pypi.org/>`_ (PyPI) is a software
repository for the Python programming language. PyPI helps you find and
install software developed and shared by the Python community.

The following guide to package pysteps was adapted from the
`PyPI <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_
official documentation.

Generating the source distribution
==================================

The first step is to generate a `source distribution
(sdist) <https://packaging.python.org/glossary/#term-source-distribution-or-sdist>`_
for the pysteps library. These are archives that are uploaded to the
`Package Index <https://pypi.org/>`_ and can be installed by pip.

To create the sdist package we need the **setuptools** package
installed.

Then, from the root folder of the pysteps source run::

   python setup.py sdist

Once this command is completed, it should generate a tar.gz (source
archive) file the **dist** directory::

   dist/
     pysteps-a.b.c.tar.gz

where a.b.c denote the version number.

Uploading the source distribution to the archive
================================================

The last step is to upload your package to the `Python Package
Index <https://pypi.org/>`_.

**Important**

Before we actually upload the distribution to the Python Index, we will
test it in `Test PyPI <https://test.pypi.org/>`_. Test PyPI is a
separate instance of the package index that allows us to try the
distribution without affecting the real index (PyPi). Because TestPyPI
has a separate database from the actual PyPI, you’ll need a separate
user account for specifically for TestPyPI. You can register your
account in https://test.pypi.org/account/register/.

Once you are registered, you can use
`twine <https://twine.readthedocs.io/en/latest/#twine-user-documentation>`_
to upload the distribution packages. Alternatively, the package can be
uploaded manually from the **Test PyPI** page.

If Twine is not installed, you can install it by running
``pip install twine`` or ``conda install twine``.


Test PyPI
^^^^^^^^^

To upload the recently created source distribution
(**dist/pysteps-a.b.c.tar.gz**) under the **dist** directory run::

   twine upload --repository-url https://test.pypi.org/legacy/ dist/pysteps-a.b.c.tar.gz

where a.b.c denote the version number.

You will be prompted for the username and password you registered with
Test PyPI. After the command completes, you should see output similar to
this::

   Uploading distributions to https://test.pypi.org/legacy/
   Enter your username: [your username]
   Enter your password:
   Uploading pysteps-a.b.c.tar.gz
   100%|█████████████████████| 4.25k/4.25k [00:01<00:00, 3.05kB/s]

Once uploaded your package should be viewable on TestPyPI, for example,
https://test.pypi.org/project/pysteps

Test the uploaded package
-------------------------

Before uploading the package to the official `Python Package
Index <https://pypi.org/>`_, test that the package can be installed
using pip.

Automatic test
^^^^^^^^^^^^^^

The simplest way to hat the package can be installed using pip is using tox
and the tox-conda plugin (conda needed).
To install these packages activate your conda development environment and run::

    conda install -c conda-forge tox tox-conda

Then, to test the installation in a minimal and an environment with all the
dependencies (full env), run::

    tox -r -e pypi_test        # Test the installation in a minimal env
    tox -r -e pypi_test_full   # Test the installation in an full env


Manual test
^^^^^^^^^^^

To manually test the installation on new environment,
create a copy of the basic development environment using the
`environment_dev.yml <https://github.com/pySTEPS/pysteps/blob/master/environment_dev.yml>`_
file in the root folder of the pysteps project::

    conda env create -f environment_dev.yml -n pysteps_test

Then we activate the environment::

    source activate pysteps_test

or::

    conda activate pysteps_test

If the environment pysteps_test was already created, remove any version of
pysteps already installed::

    pip uninstall pysteps

Now, install the pysteps package from test.pypi.org. 
Since not all the dependecies are available in the Test PyPI repository, we need to add the official repo as an extra index to pip. By doing so, pip will look first in the Test PyPI index and then in the official PyPI::

    pip install --no-cache-dir --index-url https://test.pypi.org/simple/  --extra-index-url=https://pypi.org/simple/ pysteps

To test that the installation was successful, from a folder different
than the pysteps source, run::

    pytest --pyargs pysteps


If any test didn't pass, check the sources or consider creating a new release
fixing those bugs.


Upload package to PyPi
----------------------

Once the
`sdist <https://packaging.python.org/glossary/#term-source-distribution-or-sdist>`_
package was tested, we can safely upload it to the Official PyPi
repository with::

   twine upload dist/pysteps-a.b.c.tar.gz

Now, **pysteps** can be installed by simply running::

   pip install pysteps

As an extra sanity measure, it is recommended to test the pysteps package
installed from the Official PyPi repository
(instead of the test PyPi).

Automatic test
^^^^^^^^^^^^^^

Similarly to the `Test the uploaded package`_ section, to test the
installation from PyPI in a clean environment, run::

    tox -r -e pypi

Manual test
^^^^^^^^^^^

Follow test instructions in `Test PyPI`_ section.

