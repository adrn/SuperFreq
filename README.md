SuperFreq
=========

Frequency analysis of (orbital) time series. This is close to that implemented by Monica Valluri in [NAFF](http://dept.astro.lsa.umich.edu/~mvalluri/resources.html), which itself is an implementation of the algorithm first used by Jacques Laskar (see [this review](http://arxiv.org/pdf/math/0305364v3.pdf) and citations within).

Documentation
=============

The documentation is [available here](http://adrian.pw/superfreq/).

Installation
============

You'll first need to make sure you have the required packages installed (see
pip-requirements.txt). You can use `pip` to automatically install these with

    pip install -r pip-requirements.txt

Then it's just a matter of

    python setup.py install

For the object-oriented interface to `SuperFreq`, these are the only
requirements. For more automated frequency finding, you'll also need to
install [gary](https://github.com/adrn/gary).
