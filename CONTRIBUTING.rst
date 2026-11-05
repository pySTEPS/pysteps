Contributing to pysteps
=======================

Welcome! Pysteps is a community-driven initiative for developing and
maintaining an easy to use, modular, free and open-source Python
framework for short-term ensemble prediction systems.

There are many ways to contribute to pysteps:

* contributing bug reports and feature requests
* contributing documentation
* code contributions, new features, or bug fixes
* contribute with usage examples

Who can contribute?
-----------------------

We welcome all kinds of contributions, like documentation updates, bug fixes, or new features.
The workflow for the contributions uses the usual
`GitHub pull-request flow <https://help.github.com/en/articles/github-flow>`_.

If you have ideas for new contributions to the project, feel free to get in touch with the pysteps community 
through the `project's issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.


**First Time Contributors**
-----------------------

If you are interested in helping to improve pysteps,
the best way to get started is by looking for "Good First Issue" in the
`issue tracker <https://github.com/pySTEPS/pysteps/issues>`_.

In a nutshell, the main steps to follow for contributing to pysteps are:

* Setting up the pysteps environment (see below)
* Fork and clone the repository
* Install pre-commit hooks
* Create an issue for each contribution
* Create a new branch for each contribution
* Read the Code Style guide
* Work on your changes
* Test your changes
* Push to your fork repository
* Create a new PR on the pysteps GitHub.


Setting up the pysteps environment
--------------------------------------

The recommended way to set up the developer environment is with :code:`uv`.
uv quickly installs, runs, and updates packages and their dependencies.
It also allows you to create, save, load, and switch between different environments on your local computer.

Before continuing, Mac OSX users also need to install a more recent compiler.
See instructions `here <https://pysteps.readthedocs.io/en/latest/user_guide/install_pysteps.html#install-osx-users>`__.

The developer environment can be created by syncing the optional and dev dependencies::

    uv sync --extra optional

To test if the installation went fine, you can try importing pysteps from the python interpreter by running::

    uv run python -c "import pysteps"


Fork the repository
~~~~~~~~~~~~~~~~~~~

Once you have set up the pysteps environment, the next step is creating your local copy of the repository, where you will commit your modifications.
The steps to follow are:

#. Set up Git on your computer.
#. Create a GitHub account (if you don't have one).
#. Fork the repository in your GitHub.
#. Clone a local copy of your fork. For example::

    git clone https://github.com/<your-account>/pysteps.git

Done!, now you have a local copy of pysteps git repository.
If you are new to GitHub, below you can find a list of helpful tutorials:

- http://rogerdudler.github.io/git-guide/index.html
- https://www.atlassian.com/git/tutorials

Install pre-commit hooks
~~~~~~~~~~~~~~~~~~~~~~~~

After setting up your development environment, install the git pre-commit hook by executing the following command in the repository's
root::

    pre-commit install

The pre-commit hooks are scripts executed automatically in every commit to identify simple issues with the code.
When an issue is identified (the pre-commit script exits with non-zero status), the hook aborts the commit and prints the error.
Currently, PySTEPS only tests that the code to be committed complies with :code:`bandit`, :code:`ruff format` and :code:`ruff`'s import sorter.
In case that the commit is aborted, can run these to try to find the issues::

    uv run bandit -c bandit.yaml .
    uv run ruff check
    uv run ruff format --check

This can also be done by running :code:`pre-commit run --all-files`. The latter is recommended since it indicates if the
commit contained any formatting errors (that are automatically corrected).

For more information about git hooks and the pre-commit package, see:

- https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
- https://pre-commit.com/


Create a new branch
~~~~~~~~~~~~~~~~~~~

As a collaborator, all the new contributions you want should be made in a new branch under your forked repository.

For more information on how to create and work with branches, see
`"Branches in a Nutshell" <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`__ in the Git documentation


Code Style
----------

We strongly suggest following the
`PEP8 coding standards <https://www.python.org/dev/peps/pep-0008/>`__.
Since PEP8 is a set of recommendations, these are the most important good coding practices for the pysteps project:

* Always use four spaces for indentation (don’t use tabs).
* Max line-length: 88 characters (note that we don't use the PEP8's 79 value). Enforced by `ruff`.
* Always indent wrapped code for readability. Enforced by `ruff`.
* Avoid extraneous whitespace. Enforced by `ruff`.
* Don’t use whitespace to line up assignment operators (=, :). Enforced by `ruff`.
* Avoid writing multiple statements in the same line.
* Naming conventions should follow the recommendations from
  the `Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__, summarized as follows:

   .. raw:: html

        <table rules="all" border="1" cellspacing="2" cellpadding="2">

          <tr>
            <th>Type</th>
            <th>Public</th>
            <th>Internal</th>
          </tr>

          <tr>
            <td>Packages</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Modules</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code></td>
          </tr>

          <tr>
            <td>Classes</td>
            <td><code>CapWords</code></td>
            <td><code>_CapWords</code></td>
          </tr>

          <tr>
            <td>Exceptions</td>
            <td><code>CapWords</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Functions</td>
            <td><code>lower_with_under()</code></td>
            <td><code>_lower_with_under()</code></td>
          </tr>

          <tr>
            <td>Global/Class Constants</td>
            <td><code>CAPS_WITH_UNDER</code></td>
            <td><code>_CAPS_WITH_UNDER</code></td>
          </tr>

          <tr>
            <td>Global/Class Variables</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code></td>
          </tr>

          <tr>
            <td>Instance Variables</td>
            <td><code>lower_with_under</code></td>
            <td><code>_lower_with_under</code> (protected)</td>
          </tr>

          <tr>
            <td>Method Names</td>
            <td><code>lower_with_under()</code></td>
            <td><code>_lower_with_under()</code> (protected)</td>
          </tr>

          <tr>
            <td>Function/Method Parameters</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

          <tr>
            <td>Local Variables</td>
            <td><code>lower_with_under</code></td>
            <td></td>
          </tr>

        </table>

(source: `Section 3.16.4, Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__)

- If you need to ignore part of the variables returned by a function,
  use "_" (single underscore) or __ (double underscore)::

    precip, __, metadata = import_bom_rf3('example_file.bom')
    precip2, _, metadata2 = import_bom_rf3('example_file2.bom')


- Zen of Python (`PEP 20 <https://www.python.org/dev/peps/pep-0020/>`__), the guiding principles for Python’s
  design::

    >>> import this
    The Zen of Python, by Tim Peters

    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!

For more suggestions on good coding practices for python, check these guidelines:

- `The Hitchhiker's Guide to Python <https://docs.python-guide.org/writing/style/>`__
- `Google's python style guide <http://google.github.io/styleguide/pyguide.html>`__
- `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__

**Docstrings**

Every module, function, or class must have a docstring that describe its
purpose and how to use it. The docstrings follows the conventions described in the
`PEP 257 <https://www.python.org/dev/peps/pep-0257/#multi-line-docstrings>`__
and the
`Numpy's docstrings format <https://numpydoc.readthedocs.io/en/latest/format.html>`__.

Here is a summary of the most important rules:

- Always use triple quotes for doctrings, even if it fits a single line.
- For one-line docstring, end the phrase with a period.
- Use imperative mood for all docstrings ("""Return some value.""") rather than descriptive mood
  ("""Returns some value.""").

Here is an example of a docstring::

    def adjust_lag2_corrcoef1(gamma_1, gamma_2):
        """
        A simple adjustment of lag-2 temporal autocorrelation coefficient to
        ensure that the resulting AR(2) process is stationary when the parameters
        are estimated from the Yule-Walker equations.

        Parameters
        ----------
        gamma_1 : float
          Lag-1 temporal autocorrelation coeffient.
        gamma_2 : float
          Lag-2 temporal autocorrelation coeffient.

        Returns
        -------
        out : float
          The adjusted lag-2 correlation coefficient.
        """


Contributions guidelines
------------------------

~~~~~~~~~~~~~~~~~~~~~~~~

As a collaborator, all your new contributions should be made in a new branch under your forked repository, or
if you are a member of the pySTEPS organisation, under a new branch in the pysteps repository. 

Nobody can push changes straight into the main branch; all development is done in a feature branch.
This approach has two main advantages:

- Every commit pushed to a pull request targeting ``main`` or ``v1.x`` is tested via GitHub Actions.
  If the tests fail, they do not affect the main branch.

- Once the changes are finished and the tests passed, the commits history can be squashed into a single commit and
  then merged into the main branch. Squashing the commits helps to keep a clean commit history in the main branch.

**Creating an issue**
Before you submit a contribution or fix, please open a new issue so the community can provide input. 
The discussions will help clarify the best way to approach the suggested changes and frame it in the project roadmap,
and associate it with a milestone.

AI-generated code must clearly be indicated as co-authored by AI; only a human can take responsibility for a contribution.

Once your proposed changes are ready, you need to create a pull request (PR) linked to the associated issue.
Afterward, other contributors will review your proposed changes, provide feedback in the PR discussion, and maybe,
request changes to the code, in which case the PR will need to be reviewed again. 

Copilot and other AI tools can be used as support but do not replace a review by a human.

Once the PR and all changes are approved, the PR will be merged into the main branch.

Stale PRs and issues that have had no activity for 12 months are first marked as stale and then closed.

**Important:**
Each PR should only address a single issue (e.g., fix a bug, improve documentation, etc.).
This will help to reduce the time needed to review the PR. 
For changes out of the scope of the issue, please open new issue and associated PR.


**Testing your changes**
~~~~~~~~~~~~~~~~~~~~

Before committing changes or creating pull requests, check that all the tests in the pysteps suite pass.
See the `Testing pysteps <https://pysteps.readthedocs.io/en/latest/developer_guide/test_pysteps.html#testing-pysteps>`__
for detailed instruction to run the tests.

Although it is not strictly needed, we suggest creating minimal tests for new contributions to ensure that it achieves
the desired behavior. Pysteps uses the pytest framework that it is easy to use and also supports complex functional
testing for applications and libraries.
Check the `pytests official documentation <https://docs.pytest.org/en/latest/index.html>`_ for more information.

The tests should be placed under the
`pysteps.tests <https://github.com/pySTEPS/pysteps/tree/master/pysteps/tests>`_
module.
The file should follow the **test_*.py** naming convention and have a
descriptive name.

A quick way to get familiar with the pytest syntax and the testing procedures
is checking the python scripts present in the pysteps test module.


Processing pull requests
~~~~~~~~~~~~~~~~~~~~~~~~

.. _`Squash and merge`: https://github.com/blog/2141-squash-your-commits

To process the pull request, we follow similar rules to those used in the
`mypy developer guidelines <https://github.com/python/mypy/blob/master/CONTRIBUTING.md#core-developer-guidelines>`_:

* Always wait for tests to pass before merging PRs.
* Always use "`Squash and merge`_"  to merge PRs.
* Make sure that the subject of the commit message summarizes the objective of the PR and does not finish with a dot.
* Write a new commit message before merging that provides a detailed description of the changes introduced by the PR.
  Try to keep the maximum line length under 80 characters, splitting lines if necessary.
  **IMPORTANT:** Make sure that the commit message doesn't contain the branch's commit history!
  Also, if the PR fixes an issue, mention this explicitly.
* Use the imperative mood in the subject line (e.g. "Fix typo in README").

After the PR is merged, the merged branch can be safely deleted.

Preparing a new release
~~~~~~~~~~~~~~~~~~~~~~~

Core developers should follow the steps to prepare a new release (version):

1. Before creating the actual release in GitHub, be sure that every item in the following checklist was followed:

    * In the file setup.py, update the **version="X.X.X"** keyword in the setup
      function.
    * Update the version in PKG-INFO file.
    * If new dependencies were added to pysteps since the last release, add
      them to the **environment.yml, requirements.txt**, and
      **requirements_dev.txt** files.
#. Create a new release in GitHub following
   `these guidelines <https://help.github.com/en/articles/creating-releases>`_.
   Include a detailed changelog in the release.
#. Generating the source distribution for new pysteps version and upload it to
   the `Python Package Index <https://pypi.org/>`_ (PyPI).
   See `Packaging the pysteps project <https://pysteps.readthedocs.io/en/latest/developer_guide/pypi.html#pypi-relase>`__
   for a detailed description of this process.

Owner and maintainer rules
--------------------------

* The pySTEPS organization should have exactly three owners, no more, no less.
* Being a maintainer comes with responsibilities, i.e. regularly performing tasks such as:
   - preparing, tagging and documenting releases
   - PR reviews
   - issue triage
   - organising hackathons and code sprints
   - keeping documentation and examples up to date
* After 12 months of inactivity:
   - an owner is replaced by a more active maintainer that the community agrees upon
   - a maintainer loses their role and becomes a regular member

This role can of course be restored if needed (e.g. after a sabbatical or leave).

Credits
-------

This document was based on https://opensource.guide/ and contributors guides of two Python
open-source projects:

* Py-Art_: Copyright (c) 2013, UChicago Argonne, LLC.
  `License <https://github.com/ARM-DOE/pyart/blob/master/LICENSE.txt>`_.
* mypy_: Copyright (c) 2015-2016 Jukka Lehtosalo and contributors.
  `MIT License <https://github.com/python/mypy/blob/master/LICENSE>`_.
* Official github documentation (https://help.github.com)

.. _Py-Art: https://github.com/ARM-DOE/pyart
.. _mypy: https://github.com/python/mypy

