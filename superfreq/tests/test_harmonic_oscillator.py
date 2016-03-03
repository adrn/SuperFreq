# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
try:
    import h5py
    HAS_h5py = True
except ImportError:
    HAS_h5py = False
import numpy as np
import pytest
try:
    import gary
    HAS_gary = True
except ImportError:
    HAS_gary = False

# Project
from .data.generate import get_harmonic_oscillator_orbits
from ..naff import SuperFreq # , compute_actions

@pytest.mark.skipif(not HAS_h5py or not HAS_gary,
                    reason='h5py and gary must be installed to run this test')
def test_frequencies():
    n_orbits = 4
    cache_file = get_harmonic_oscillator_orbits(n_orbits=n_orbits)

    with h5py.File(cache_file, 'r') as f:
        all_t = f['orbits']['t'][:]
        all_x = f['orbits']['x'][:]
        all_v = f['orbits']['v'][:]
        initial_freqs = f['initial']['freqs'][:]

    for n in range(n_orbits):
        # run superfreq
        sf = SuperFreq(all_t[:,n])
        true_freq = initial_freqs[:,n]

        fs = [(all_x[i,:,n] + 1j*all_v[i,:,n]) for i in range(2)]
        freqs,tbl,ixes = sf.find_fundamental_frequencies(fs, nintvec=10)

        np.testing.assert_allclose(-freqs, true_freq, rtol=1E-7)

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
#     import gary.potential as gp
#     from gary.integrate import DOPRI853Integrator
#     from gary.units import galactic

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

