#########
SuperFreq
#########

.. image:: _static/rick_james.jpg
    :width: 196px
    :align: center

|
|

Introduction
============

A Python package for orbital frequency analysis.

This package implements a version of the Numerical Analysis of Fundamental Frequencies close to that implemented by Monica Valluri in `NAFF <http://dept.astro.lsa.umich.edu/~mvalluri/resources.html>`_, which itself is an implementation of the algorithm first used by Jacques Laskar (see `this review <http://arxiv.org/pdf/math/0305364v3.pdf>`_ and citations within).

The package is developed in
`a public repository on GitHub <https://github.com/adrn/SuperFreq>`_. If you
have any trouble installing, find any bugs, or have any questions, please `open an issue on GitHub <https://github.com/adrn/SuperFreq/issues>`_.

Getting started
===============

SuperFreq operates on time series, e.g., orbits. This package does not provide any functionality for integrating orbits or transforming between coordinate representations (for that, you might want to install `astropy <https://github.com/astropy/astropy>`_ and `gary <https://github.com/adrn/gary>`_).

Reference / API
===============

.. autosummary::
   :nosignatures:
   :toctree: _superfreq/

   superfreq.SuperFreq
