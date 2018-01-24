SuperFreq
=========

Frequency analysis of orbital time series.

|Build Status|

This is close to that implemented by Monica Valluri in `NAFF`_, which
itself is an implementation of the algorithm first used by Jacques
Laskar (see `this review`_ and citations within).


Documentation
-------------

The documentation is `available here`_.


Installation
------------

``SuperFreq`` is easily installed via ``pip``. To install the latest stable
version:

.. code:: bash

    pip install superfreq

To install the latest development version:

.. code:: bash

    pip install git+https://github.com/adrn/superfreq

For the object-oriented interface to ``SuperFreq``, these are the only
requirements. For more automated frequency finding, you’ll also need to
install `gala`_.

.. _NAFF: http://dept.astro.lsa.umich.edu/~mvalluri/resources.html
.. _this review: http://arxiv.org/pdf/math/0305364v3.pdf
.. _available here: http://superfreq.readthedocs.io/
.. _Gala: https://github.com/adrn/gala
.. _downloading the source: https://github.com/adrn/SuperFreq/archive/master.zip
.. _gala: https://github.com/adrn/gala

.. |Build Status| image:: https://travis-ci.org/adrn/SuperFreq.svg
   :target: https://travis-ci.org/adrn/SuperFreq
