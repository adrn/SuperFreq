# coding: utf-8

""" Test NAFF stuff """

from __future__ import division, print_function

import numpy as np

def test_cy_naff():
    from ..naff import naff_frequency, SuperFreq

    t = np.linspace(0., 300., 12000)
    naff = SuperFreq(t)

    true_ws = 2*np.pi*np.array([0.581, 0.73])
    true_as = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                        1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
    f = np.sum(true_as[None] * np.exp(1j * true_ws[None] * t[:,None]), axis=1)

    ww = naff_frequency(true_ws[0], naff.tz, naff.chi,
                        np.ascontiguousarray(f.real),
                        np.ascontiguousarray(f.imag),
                        naff.T)
    np.testing.assert_allclose(ww, true_ws[0], atol=1E-8)

# TODO: make a class to do these tests for given arrays of freqs, complex amps.
