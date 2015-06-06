# coding: utf-8

"""
    This test requires some data generated using Gary. The data is not stored
    in the repository so you will have to generate this if you want to run
    this test suite. Generating the test data requires both Gary and
    HDF5 / h5py. The script to generate the data can be run with::

        python superfreq/tests/data/generate.py

"""

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import h5py
import numpy as np
import pytest

# Project
from ..naff import SuperFreq

# TODO: wat do about this
# cache_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
# cache_file = os.path.join(cache_path, "isochrone_orbits.h5")
cache_file = "/Users/adrian/projects/superfreq/superfreq/tests/data/isochrone_orbits.h5"
HAS_DATA = os.path.exists(cache_file)

@pytest.mark.skipif('not HAS_DATA')
def test_orbits():
    f = h5py.File(cache_file, 'r')
    t = f['orbits']['t']
    w = f['orbits']['w']

    for n in range(w.shape[1]):
        # run superfreq
        sf = SuperFreq(t[:,n])
        true_freq = f['truth']['freqs'][n]

        # ww = cartesian_to_polar(w[:,n].copy())
        ww = w[:,n].copy()
        fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
        freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=10)

        freq_rtheta = np.array([freqs[0] + freqs[1], freqs[1]])
        np.testing.assert_allclose(np.abs(freq_rtheta), true_freq, atol=1E-8)
