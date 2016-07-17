SuperFreq
=========

Frequency analysis of (orbital) time series.

[![Build Status](https://travis-ci.org/adrn/SuperFreq.svg)](https://travis-ci.org/adrn/SuperFreq)

This is close to that implemented by Monica Valluri in [NAFF](http://dept.astro.lsa.umich.edu/~mvalluri/resources.html), which itself is an implementation of the algorithm first used by Jacques Laskar (see [this review](http://arxiv.org/pdf/math/0305364v3.pdf) and citations within).

Documentation
-------------

The documentation is [available here](http://adrian.pw/superfreq/).

Dependencies
------------

- Python >= 2.7.9
- Numpy >= 1.8
- Cython >= 0.21

Installation
------------

`SuperFreq` is easily installed via `pip`:

```bash
pip install superfreq
```

You could also install from the source by cloning or [downloading the source](https://github.com/adrn/SuperFreq/archive/master.zip) from this repository. You'll first need to make sure you have the required packages installed --- you can use `pip` to automatically install these with:

```bash
pip install -r pip-requirements.txt
```

Then run

```bash
python setup.py install
```

For the object-oriented interface to `SuperFreq`, these are the only requirements. For more automated frequency finding, you'll also need to install [gala](https://github.com/adrn/gala).
