#################
``micromodelsim``
#################

``micromodelsim`` (for "microstructural model simulator") is a Python package
for generating diffusion-weighted nuclear magnetic resonance (NMR) signals from
microstructural models. It is designed to help in the development and validation
of microstructural models and to enable efficient training of machine learning
models for parameter estimation.

Installation
============

The package can be installed with `pip <https://github.com/pypa/pip>`_:

..  code-block::

    pip install git+https://github.com/kerkelae/micromodelsim.git

If you have a CUDA-capable graphical processing unit (GPU), it is recommended
to also install `JAX <https://jax.readthedocs.io/>`_ for greatly improved
performance. JAX installation instructions are provided `here
<https://github.com/google/jax#installation>`_.

Getting started
===============

The `tutorial <https://micromodelsim.rtfd.io/en/latest/tutorial.html>`_ shows
how to use the package. Details can be found in the `reference
<https://micromodelsim.rtfd.io/en/latest/reference.html>`_.
