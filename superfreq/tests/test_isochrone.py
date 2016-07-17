# coding: utf-8

"""
    This test requires some data generated using Gala. The data is not stored
    in the repository so you will have to generate this if you want to run
    this test suite. Generating the test data requires both Gala and
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
from .helpers import cartesian_to_poincare
from ..naff import SuperFreq, compute_actions

# TODO: wat do about this
# cache_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
# cache_file = os.path.join(cache_path, "isochrone_orbits.h5")
cache_file = "/Users/adrian/projects/superfreq/superfreq/tests/data/isochrone_orbits.h5"
HAS_DATA = os.path.exists(cache_file)

@pytest.mark.skipif('not HAS_DATA')
def test_frequencies():
    f = h5py.File(cache_file, 'r')
    t = f['orbits']['t']
    w = f['orbits']['w']

    for n in range(w.shape[1])[:10]:
        # run superfreq
        sf = SuperFreq(t[:,n])
        true_freq = f['truth']['freqs'][n]

        ww = cartesian_to_poincare(w[:,n].copy())
        fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
        freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=10)

        np.testing.assert_allclose(np.abs(freqs), true_freq, atol=1E-8)

# @pytest.mark.skipif('not HAS_DATA')
# def test_actions():
#     f = h5py.File(cache_file, 'r')
#     t = f['orbits']['t']
#     w = f['orbits']['w']

#     for n in range(w.shape[1]):
#         # run superfreq
#         sf = SuperFreq(t[:,n])
#         true_actn = f['truth']['actions'][n]

#         ww = w[:,n].copy()
#         fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
#         freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=30)

#         Js = compute_actions(freqs, tbl, max_int=12)
#         print("SuperFreq", Js)
#         print("True", true_actn)
#         break
#         # print(np.abs((true_actn - Js) / true_actn))
#         # np.testing.assert_allclose(np.abs(freqs), true_actn, rtol=1E-1)

# def test_single_orbit_actions():
#     import gala.potential as gp
#     from gala.integrate import DOPRI853Integrator
#     from gala.units import galactic

#     w0 = np.array([15.,0,0,0,0.12,0.])
#     pot = gp.IsochronePotential(m=1E11, b=5., units=galactic)
#     t,w = pot.integrate_orbit(w0, dt=0.5, nsteps=72000, Integrator=DOPRI853Integrator)

#     sf = SuperFreq(t, p=1)

#     # ww = cartesian_to_poincare(w[:,0].copy())
#     ww = w[:,0].copy()
#     fs = [(ww[:,i] + 1j*ww[:,i+2]) for i in range(2)]
#     freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=30, break_condition=None)

#     from ..naff import find_integer_vectors
#     imax = 12
#     nvecs = find_integer_vectors(freqs, tbl, max_int=imax)
#     amp2 = np.zeros([2*imax+1]*2)
#     for nvec,row in zip(nvecs, tbl):
#         slc = [slice(x+imax,x+imax+1,None) for x in nvec]
#         amp2[slc] += row['A'].real**2

#     Js = np.zeros(2)
#     for nvec in nvecs:
#         Js += nvec * nvec.dot(freqs) * amp2[nvec[0]+imax, nvec[1]+imax]
#     print(Js)

#     # for i in [0,1]:
#     #     _tbl = tbl[tbl['idx'] == i]
#     #     nvecs = find_integer_vectors(freqs, _tbl)
#     #     for nvec,omega_k,amp in zip(nvecs, _tbl['freq'], _tbl['A']):
#     #         Js[i] += nvec[i] * omega_k * amp.real**2

#     # print(Js)
#     # print(freqs)

#     print(compute_actions(freqs, tbl, max_int=12))

#     true_J,_,_ = pot.action_angle(w0[:3], w0[3:])
#     print(true_J)

