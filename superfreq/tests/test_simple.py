# coding: utf-8

""" Simple unit tests of SuperFreq """

# Third-party
from astropy.utils import isiterable
import numpy as np

# Project
from ..naff import SuperFreq


def test_cy_naff():
    """
    This checks the Cython frequency determination function. We construct a simple
    time series with known frequencies and amplitudes and just verify that the
    strongest frequency pulled out by NAFF is correct.
    """

    from .._naff import naff_frequency

    t = np.linspace(0., 300., 12000)
    true_ws = 2*np.pi*np.array([0.581, 0.73])
    true_as = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                        1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])

    for p in range(1,4+1): # try different filter exponents, p
        ff = SuperFreq(t, p=p)
        for sign in [1.,-1.]: # make sure we recover the correct sign of the frequency
            true_omegas = true_ws * sign
            f = np.sum(true_as[None] * np.exp(1j * true_omegas[None] * t[:,None]), axis=1)

            ww = naff_frequency(true_omegas[0], ff.tz, ff.chi,
                                np.ascontiguousarray(f.real),
                                np.ascontiguousarray(f.imag),
                                ff.T)

            np.testing.assert_allclose(ww, true_omegas[0], atol=1E-8)
'''
def test_cy_naff_scaling():
    """
    This plots how the accuracy in frequency, angle, and phase recovery scales with
    a) the number of timesteps
    b) the length of the time window
    """

    from .._naff import naff_frequency

    true_periods = np.array([1.556, 1.7211])
    true_ws = 2*np.pi/true_periods
    true_as = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                        1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])

    length_grid = np.round(2**np.arange(4,12+1,0.1)).astype(int)
    size_grid = np.round(2**np.arange(4,12+1,0.1)).astype(int)
    shp = (length_grid.size, size_grid.size)
    xgrid,ygrid = np.meshgrid(length_grid, size_grid)
    grid = np.vstack((np.ravel(xgrid), np.ravel(ygrid))).T

    p = 1
    rel_errs = []
    for length,size in grid:
        print(length, size)
        t = np.linspace(0., length, size)

        ff = SuperFreq(t, p=p)
        f = np.sum(true_as[None] * np.exp(1j * true_ws[None] * t[:,None]), axis=1)

        ww = naff_frequency(true_ws[0], ff.tz, ff.chi,
                            np.ascontiguousarray(f.real),
                            np.ascontiguousarray(f.imag),
                            ff.T)

        rel_errs.append(np.abs(true_ws[0] - ww) / true_ws[0])
    rel_errs = np.array(rel_errs)
    print(rel_errs.shape, shp)

    # --

    import matplotlib.pyplot as pl
    l_xgrid, l_ygrid = np.log2(xgrid), np.log2(ygrid)
    dx = l_xgrid[0,1]-l_xgrid[0,0]
    dy = l_ygrid[1,0]-l_ygrid[0,0]

    # pl.pcolor(np.log2(xgrid), np.log2(ygrid),
    #           np.log2(rel_errs.reshape(xgrid.shape)), cmap='viridis')
    pl.imshow(np.log10(rel_errs.reshape(xgrid.shape)), cmap='viridis', interpolation='nearest',
              extent=[l_xgrid.min()-dx/2, l_xgrid.max()+dx/2,
                      l_ygrid.min()-dy/2, l_ygrid.max()+dy/2],
              origin='bottom', vmin=-12, vmax=-1)
    pl.xlabel('Window length')
    pl.ylabel('Num. timesteps')
    pl.colorbar()
    pl.gca().set_aspect('equal')
    pl.show()
'''

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

            if not isiterable(self.p):
                ps = [self.p]
            else:
                ps = self.p

            for p in ps:
                print(i, p)
                # create SuperFreq object for this time array
                sf = SuperFreq(t, p=p)

                # solve for the frequencies
                w,amp,phi = sf.frecoder(f[:sf.n], break_condition=1E-5)

                np.testing.assert_allclose(self.omega, w[:nfreq], rtol=1E-7)
                np.testing.assert_allclose(self.A, amp[:nfreq], rtol=1E-5)
                np.testing.assert_allclose(self.phi, phi[:nfreq], rtol=1E-3)

    def test_rolling_window(self):
        ts = [np.linspace(0.+dd, 100.+dd, 10000) for dd in np.linspace(0,20,64)]
        for i,t in enumerate(ts):
            logger.debug(i, t.min(), t.max(), len(t))
            f = self.make_f(t)
            nfreq = len(self.omega)

            if not isiterable(self.p):
                ps = [self.p]
            else:
                ps = self.p

            for p in ps:
                print(i, p)

                # create SuperFreq object for this time array
                sf = SuperFreq(t, p=p)

                # try recovering the strongest frequency
                w,amp,phi = sf.frecoder(f[:sf.n], break_condition=1E-5)

                np.testing.assert_allclose(self.omega, w[:nfreq], rtol=1E-7)
                np.testing.assert_allclose(self.A, amp[:nfreq], rtol=1E-5)
                np.testing.assert_allclose(self.phi, phi[:nfreq], rtol=1E-4)

class TestSimple1(SimpleBase):
    omega = 2*np.pi*np.array([0.581])
    amp = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.)))])
    p = 4

class TestSimple2(SimpleBase):
    omega = 2*np.pi*np.array([0.581, 0.73])
    amp = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                    1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.)))])
    p = 4

class TestSimple3(SimpleBase):
    omega = 2*np.pi*np.array([0.581, 0.73, 0.113])
    amp = np.array([5*(np.cos(np.radians(15.)) + 1j*np.sin(np.radians(15.))),
                    1.8*(np.cos(np.radians(85.)) + 1j*np.sin(np.radians(85.))),
                    0.7*(np.cos(np.radians(45.)) + 1j*np.sin(np.radians(45.)))])
    p = 4
