############
Contributing
############

Contributions to the package from other developers and scientists are welcome!
Note that a basic understanding of Python software development is required.

Development workflow
====================

1. Fork the `repository on GitHub
   <https://github.com/kerkelae/micromodelsim/>`_.

2. Clone your fork.

.. tip::

   When making changes to the code, it is helpful to install the package in
   editable mode by executing the following in the root directory of the
   repository:

   .. code-block::

        pip install -e .

3. Create a branch with a descriptive name.

4. Write code, commit changes with clear messages, and push the changes to
   your fork.

5. Open a pull request on GitHub to have your code reviewed and merged into the
   main branch.

GitHub docs provide more information on `forking a repository
<https://docs.github.com/en/get-started/quickstart/fork-a-repo>`_ and `creating
pull requests
<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/
proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-
a-fork>`_.

Code style
==========

All code should be formatted with `Black <https://github.com/psf/black>`_ using
the default settings and documented following the `NumPy style guide
<https://numpydoc.readthedocs.io/en/latest/format.html>`_.

Tests
=====

It is important to run the automated tests to ensure that your changes have not
broken anything. The tests require `DIPY <https://dipy.org>`_ and
`pytest <https://pytest.org/>`_. The tests can be run by executing the
following in the root directory of the repository:

.. code-block::

    pytest

Documentation
=============

Building the documentation locally requires
`make <https://www.gnu.org/software/make/>`_, `pandoc <https://pandoc.org/>`_,
`sphinx <https://www.sphinx-doc.org/>`_,
`nbsphinx <https://nbsphinx.readthedocs.io/>`_, and
`furo <https://pradyunsg.me/furo/>`_. Documentation can be built locally by
executing the following in the ``docs`` directory:

.. code-block::

    make clean
    make html

This will generate the documentation in ``docs/_build/html``.

Once the changes are merged into the main branch, the documentation is
automatically updated online.
