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
from ..naff import SuperFreq, compute_actions

# TODO: wat do about this
# cache_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
# cache_file = os.path.join(cache_path, "isochrone_orbits.h5")
cache_file = "/Users/adrian/projects/superfreq/superfreq/tests/data/isochrone_orbits.h5"
HAS_DATA = os.path.exists(cache_file)

def cartesian_to_polar(w):
    # assuming z == 0
    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+2] + w[...,1]*w[...,1+2]) / R
    vPhi = w[...,0]*w[...,1+2] - w[...,1]*w[...,0+2]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    new_w = np.vstack((R.T, pp_phi.T, vR.T, pp_phidot.T)).T
    return new_w

@pytest.mark.skipif('not HAS_DATA')
def test_isochrone():
    f = h5py.File(cache_file, 'r')
    t = f['orbits']['t']
    w = f['orbits']['w']

    for n in range(w.shape[1]):
        # run superfreq
        sf = SuperFreq(t[:,n])

        true_actn = f['truth']['actions'][n]
        true_angl = f['truth']['angles'][n]
        true_freq = f['truth']['freqs'][n]

        # import gary.dynamics as gd
        # import matplotlib.pyplot as plt
        # fig = gd.plot_orbits(w[:,n], marker=None)
        # plt.show()
        # return

        ww = cartesian_to_polar(w[:,n].copy())
        fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
        freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=10)

        np.testing.assert_allclose(np.abs(freqs), true_freq, atol=1E-8)
        # print("Isochrone: {0}".format(true_freq))
        # print("SuperFreq: {0}".format(freqs))

        sf_acts = compute_actions(freqs, tbl)
        print("Isochrone act: {0}".format(true_actn))
        print("SuperFreq act: {0}".format(sf_acts))
