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

# @pytest.mark.skipif('not HAS_DATA')
@pytest.mark.skipif(True)
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

        return

        # SUPAH TESTIN
        from ..naff import find_integer_vectors

        ww = w[:,n].copy()
        fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
        freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=30)

        slc = [slice(-15,15+1,None)]*len(freqs)
        all_nvecs = np.vstack(np.vstack(np.mgrid[slc].T))


        _tbl = tbl[tbl['idx'] == 0]
        nvecs = find_integer_vectors(freqs, _tbl)
        f_x_t = 0.
        for row,nvec in zip(_tbl, nvecs):
            f_x_t += row['A'] * np.exp(1j * nvec.dot(freqs) * t[:,n])

        import matplotlib.pyplot as plt
        plt.plot(w[:,n,0], f_x_t.real, marker=None)
        plt.show()
        return

        nvecs = find_integer_vectors(freqs, tbl)

        Js = np.zeros(2)
        for j in np.unique(tbl['idx']):
            _tbl = tbl[tbl['idx'] == j]
            _nvecs = nvecs[tbl['idx'] == j]
            for i in range(len(_tbl)):
                row = _tbl[i]
                nvec = _nvecs[i]
                ix = (nvecs[:,0] == nvec[0]) & (nvecs[:,1] == nvec[1])

                A = np.sum(tbl['|A|'].real[ix]**2)
                Js[j] += nvec[j] * nvec.dot(freqs) * A
                # Js[j] += nvec[j] * nvec.dot(freqs) * row['|A|']**2
                # Js[j] += _nvecs[i,j] * _nvecs[i].dot(freqs) * row['A'].real**2

        print("Isochrone act: {0}".format(true_actn))
        print(Js)
        return

        print("Isochrone act: {0}".format(true_actn))
        print("SuperFreq act: {0}".format(sf_acts))

        break
