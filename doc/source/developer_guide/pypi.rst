.. _pypi_relase:

=============================
Packaging the PySteps project
=============================

The `Python Package Index <https://pypi.org/>`__ (PyPI) is a software
repository for the Python programming language. PyPI helps you find and
install software developed and shared by the Python community.

The following guide to package PySteps was adapted from the
`PyPI <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`__
official documentation.

Generating the source distribution
==================================

The first step is to generate a `source distribution
(sdist) <https://packaging.python.org/glossary/#term-source-distribution-or-sdist>`__
for the PySTEPS library. These are archives that are uploaded to the
`Package Index <https://pypi.org/>`__ and can be installed by pip.

To create the the sdist package we need the **setuptools** package
installed.

Then, from the root folder of the PySTEPS source run::

   python setup.py sdist

Once this command is completed, it should generate a tar.gz (source
archive) file the **dist** directory::

   dist/
     pysteps-1.0.0.tar.gz

Uploading the source distribution to the archive
================================================

The last step is to upload your package to the `Python Package
Index <https://pypi.org/>`__.

**Important**

Before we actually upload the distribution to the Python Index, we will
test it in `Test PyPI <https://test.pypi.org/>`__. Test PyPI is a
separate instance of the package index that allows us to try the
distribution without affecting the real index (PyPi). Because TestPyPI
has a separate database from the actual PyPI, you’ll need a separate
user account for specifically for TestPyPI. You can register your
account in https://test.pypi.org/account/register/.

Once you are registered, you can use
`twine <https://twine.readthedocs.io/en/latest/#twine-user-documentation>`__
to upload the distribution packages. Alternatively, the package can be
uploaded manually from the **Test PyPI** page.

If Twine is not installed, you can install it by running
``pip install twine`` or ``conda install twine``.

To upload the recently created source distribution
(**dist/pysteps-1.0.0.tar.gz**) under the **dist** directory run::

   twine upload --repository-url https://test.pypi.org/legacy/ dist/pysteps-1.0.0.tar.gz

You will be prompted for the username and password you registered with
Test PyPI. After the command completes, you should see output similar to
this::

   Uploading distributions to https://test.pypi.org/legacy/
   Enter your username: [your username]
   Enter your password:
   Uploading pysteps-1.0.0.tar.gz
   100%|█████████████████████| 4.25k/4.25k [00:01<00:00, 3.05kB/s]

Once uploaded your package should be viewable on TestPyPI, for example,
https://test.pypi.org/project/pysteps

Test the uploaded package
-------------------------

Before uploading the package to the official `Python Package
Index <https://pypi.org/>`__, test that the package can be installed
using pip by running::

   pip install --index-url https://test.pypi.org/simple/ pysteps

To test that the installation was successful, from a folder different
than the pysteps source, run the `pysteps’s test
suite <https://github.com/pySTEPS/pysteps/wiki/Testing-pysteps>`__::

   pytest --pyargs pysteps

Uploaded package to the Official PyPi
-------------------------------------

Once the
`sdist <https://packaging.python.org/glossary/#term-source-distribution-or-sdist>`__
package was tested, we can safely upload it to the Official PyPi
repository with::

   twine upload dist/pysteps-1.0.0.tar.gz

Now, **pysteps** can be installed by simply running::

   pip install pysteps
