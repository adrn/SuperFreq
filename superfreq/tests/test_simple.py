# coding: utf-8

""" Simple unit tests of SuperFreq """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Third-party
import numpy as np

# Project
from ..naff import SuperFreq

def test_cy_naff():
    """ This checks the Cython frequency determination function """

    from .._naff import naff_frequency

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

class SimpleBase(object):

    """ Need to define:

            self.amp
            self.omega
            self.p

        in subclass setup().
    """
    def setup(self):
        self.A = np.sqrt(self.amp.imag**2 + self.amp.real**2)
        self.phi = np.arctan2(self.amp.imag, self.amp.real)

    def make_f(self, t):
        a = self.amp
        w = self.omega
        return np.sum(a[None] * np.exp(1j * w[None] * t[:,None]), axis=1)

    def test_freq_recovery(self):
        # define a bunch of arrays of times to make sure SuperFreq isn't
        #   sensitive to the times
        ts = [np.linspace(0., 150., 12000),
              np.linspace(0., 150., 24414),
              np.linspace(0., 150., 42104),
              np.linspace(150., 300., 12000),
              np.linspace(150., 300., 24414),
              np.linspace(150., 300., 42104),
              np.linspace(0., 150., 12000) + 50*(2*np.pi/self.omega[0])]

        for i,t in enumerate(ts):
            logger.debug(i, t.min(), t.max(), len(t))
            f = self.make_f(t)
            nfreq = len(self.omega)

            # create SuperFreq object for this time array
            sf = SuperFreq(t, p=self.p)

            # solve for the frequencies
            w,amp,phi = sf.frecoder(f[:sf.n], break_condition=1E-5)
            np.testing.assert_allclose(self.omega, w[:nfreq], atol=1E-8)
            np.testing.assert_allclose(self.A, amp[:nfreq], atol=1E-6)
            np.testing.assert_allclose(self.phi, phi[:nfreq], atol=1E-6)

    def test_rolling_window(self):
        ts = [np.linspace(0.+dd, 150.+dd, 42104) for dd in np.linspace(0,10,50)]
        dws = []
        for i,t in enumerate(ts):
            logger.debug(i, t.min(), t.max(), len(t))
            f = self.make_f(t)
            nfreq = len(self.omega)

            # create SuperFreq object for this time array
            sf = SuperFreq(t, p=self.p)

            # try recovering the strongest frequency
            w,amp,phi = sf.frecoder(f[:sf.n], break_condition=1E-5)
            dws.append(np.abs(self.omega - w[:nfreq]))

        dws = np.array(dws)
        assert np.all(np.abs(dws) < 1E-8)

class TestSimple1(SimpleBase):

    def setup(self):
        self.omega = 2*np.pi*np.array([0.581, 0.73])
        self.amp = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                             1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
        self.p = 2
        super(TestSimple1, self).setup()

class TestSimple2(SimpleBase):

    def setup(self):
        self.omega = 2*np.pi*np.array([0.581, -0.73, 0.91])
        self.amp = np.array([5*(np.cos(np.radians(35.)) + 1j*np.sin(np.radians(35.))),
                             1.8*(np.cos(np.radians(75.)) + 1j*np.sin(np.radians(75.))),
                             0.7*(np.cos(np.radians(45.)) + 1j*np.sin(np.radians(45.)))])
        self.p = 2
        super(TestSimple2, self).setup()
