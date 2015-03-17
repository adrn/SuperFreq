# coding: utf-8

""" Port of NAFF to Python """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import time

# Third-party
from astropy import log as logger
import numpy as np
from numpy.fft import fftfreq
# try:
#     import pyfftw
#     HAS_PYFFTW = True
# except ImportError:
#     from numpy.fft import fft
#     HAS_PYFFTW = False
from numpy.fft import fft
HAS_PYFFTW = False
# TODO: enable and fix PyFFTW support -- current implementation is broken / gives wrong phi(w)

# Project
from .core import classify_orbit, align_circulation_with_z, check_for_primes
from ._naff import naff_frequency
from ..integrate.simpsgauss import simpson

__all__ = ['NAFF', 'poincare_polar', 'orbit_to_freqs']

def hanning(x):
    return 1 + np.cos(x)

def poincare_polar(w):
    ndim = w.shape[-1]//2

    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    # phi = np.arctan2(w[...,1], w[...,0])
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+ndim] + w[...,1]*w[...,1+ndim]) / R
    vPhi = w[...,0]*w[...,1+ndim] - w[...,1]*w[...,0+ndim]

    fs = []
    fs.append(R + 1j*vR)
    fs.append(np.sqrt(2*np.abs(vPhi))*(np.cos(phi) + 1j*np.sin(phi)))  # PP
    fs.append(w[...,2] + 1j*w[...,2+ndim])

    return fs

class NAFF(object):
    """
    Implementation of the Numerical Analysis of Fundamental Frequencies (NAFF)
    method of Laskar, later modified by Valluri and Merritt (see references below).

    This algorithm attempts to numerically find the fundamental frequencies of an
    input orbit (time series) and can also find approximate actions for the orbit.
    The basic idea is to Fourier transform the orbit convolved with a Hanning filter,
    find the most significant peak, subtract that frequency, and iterate on this
    until convergence or for a fixed number of terms. The fundamental frequencies
    can then be solved for by assuming that the frequencies found by the above method
    are integer combinations of the fundamental frequencies.

    For more information, see:

        - Laskar, J., Froeschlé, C., and Celletti, A. (1992)
        - Laskar, J. (1993)
        - Papaphilippou, Y. and Laskar, J. (1996)
        - Valluri, M. and Merritt, D. (1998)

    Parameters
    ----------
    t : array_like
        Array of times.
    keep_calm : bool (optional)
        If something fails when solving for the frequency of a given component,
        ``keep_calm`` determines whether to throw a RuntimeError or exit gracefully.
        If set to ``True``, will exit quietly and carry on.

    """

    def __init__(self, t, keep_calm=True):

        n = len(t)
        self.n = check_for_primes(n)

        if self.n != len(t):
            logger.info("Truncating time series to length={0} to avoid large prime divisors."
                        .format(self.n))

        # array of times
        self.t = t[:self.n]

        # average time
        t_avg = 0.5 * (self.t[-1] + self.t[0])

        # re-center time so middle is 0
        self.tz = self.t - t_avg
        self.dt = np.abs(self.tz[1] - self.tz[0])

        # time window size: time series goes from -T to T
        self.T = 0.5 * (self.t[-1] - self.t[0])

        # pre-compute values of Hanning filter for this window
        self.chi = hanning(self.tz * np.pi/self.T)  # the argument is 2π/(2T)

        # when solving for frequencies and removing components from the time series,
        #   if something fails for a given component and keep_calm is set to True,
        #   NAFF will exit gracefully instead of throwing a RuntimeError
        self.keep_calm = keep_calm

    def frequency(self, f):
        """
        Find the most significant frequency of a (complex) time series, :math:`f(t)`,
        by Fourier transforming the function convolved with a Hanning filter and
        picking the most significant peak. This assumes the time series, `f`,
        is aligned with / given at the times specified when constructing this
        object. An internal function.

        Parameters
        ----------
        f : array_like
            Complex time-series, :math:`q(t) + i p(t)`.

        Returns
        -------
        freq : numeric
            The strongest frequency in the specified complex time series, ``f``.

        """

        if len(f) != self.n:
            logger.warning("Truncating time series to match shape of time array ({0}) ({1})"
                           .format(len(f), self.n))
            f = f[:self.n]

        # take Fourier transform of input (complex) function f
        # if HAS_PYFFTW:
        #     _f = pyfftw.n_byte_align_empty(f.size, 16, 'complex128')
        #     _f[:] = f

        #     fft_obj = pyfftw.builders.fft(f, overwrite_input=True,
        #                                   planner_effort='FFTW_ESTIMATE')
        #     fff = fft_obj() / np.sqrt(self.n)
        # else:
        t1 = time.time()
        fff = fft(f) / np.sqrt(self.n)
        logger.log(0, "Took {} seconds to FFT.".format(time.time()-t1))

        # frequencies
        omegas = 2*np.pi*fftfreq(f.size, self.dt)

        # wmax is just an initial guess for optimization
        xyf = np.abs(fff)
        wmax = xyf.argmax()
        if np.allclose(xyf[wmax], 0):
            # return early -- "this may be an axial or planar orbit"
            logger.log(0, "Returning early - may be an axial or planar orbit?")
            return 0.

        # real and complex part of input time series
        Re_f = f.real.copy()
        Im_f = f.imag.copy()

        # frequency associated with the peak index
        omega0 = omegas[wmax]

        freq = naff_frequency(omega0, self.tz, self.chi, Re_f, Im_f, self.T)
        return freq

    def frecoder(self, f, nintvec=12, break_condition=1E-7):
        """
        For a given number of iterations, or until the break condition is met,
        solve for strongest frequency of the input time series, then subtract
        it from the time series.

        This function is meant to be the same as the subroutine FRECODER in
        Monica Valluri's Fortran NAFF routines.

        Parameters
        ----------
        f : array_like
            Complex time-series, :math:`q(t) + i p(t)`.
        nintvec : int (optional)
            Number of integer vectors to find or number of frequencies to find and subtract.
        break_condition : numeric (optional)
            Break the iterations of the time series maximum value or amplitude of the
            subtracted frequency is smaller than this value. Set to ``None`` if you want
            to always iterate for `nintvec` frequencies.

        Returns
        -------
        omega : :class:`numpy.ndarray`
            Array of frequencies for each component in the time series.
        ampl : :class:`numpy.ndarray`
            Array of real amplitudes for each component in the time series.
        phi : :class:`numpy.ndarray`
            Array of phases for the complex amplitudes for each component
            in the time series.
        """

        # initialize container arrays
        ecap = np.zeros((nintvec,len(self.t)), dtype=np.complex64)
        omega = np.zeros(nintvec)
        A = np.zeros(nintvec)
        phi = np.zeros(nintvec)

        fk = f.copy()
        logger.debug("-"*50)
        logger.debug("k    ωk    Ak    φk(deg)    ak")
        broke = False
        for k in range(nintvec):
            try:
                omega[k] = self.frequency(fk)
            except RuntimeError:
                if self.keep_calm:
                    broke = True
                    break
                else:
                    raise

            if k == 0:
                # compute exp(iωt) for first frequency
                ecap[k] = np.exp(1j*omega[k]*self.t)
            else:
                ecap[k] = self.gso(ecap, omega[k], k)

            # get complex amplitude by projecting exp(iωt) on to f(t)
            ab = self.hanning_product(fk, ecap[k])
            A[k] = np.abs(ab)
            phi[k] = np.arctan2(ab.imag, ab.real)

            # remove the new orthogonal frequency component from the f_k
            fk -= ab*ecap[k]

            logger.debug("{}  {:.6f}  {:.6f}  {:.2f}  {:.6f}"
                         .format(k,omega[k],A[k],np.degrees(phi[k]),ab))

            if break_condition is not None and A[k] < break_condition:
                broke = True
                break

        if broke:
            return omega[:k], A[:k], phi[:k]
        else:
            return omega[:k+1], A[:k+1], phi[:k+1]

    def hanning_product(self, u1, u2):
        r"""
        Compute the scalar product of two 'vectors', `u1` and `u2`.
        The scalar product is defined with the Hanning filter as

        .. math::

            <u_1, u_2> = \frac{1}{2 T} \int \, u_1(t) \, \chi(t) \, u_2^*(t)\,dt

        Parameters
        ----------
        u1 : array_like
        u2 : array_like

        Returns
        -------
        prod : float
            Scalar product.
        """

        # First find complex conjugate of vector u2 and construct integrand
        integ = u1 * np.conj(u2) * self.chi
        integ_r = np.ascontiguousarray(integ.real)
        integ_i = np.ascontiguousarray(integ.imag)

        # Integrate the real part
        real = simpson(integ_r, self.dt)

        # Integrate Imaginary part
        imag = simpson(integ_i, self.dt)

        return (real + 1j*imag) / (2.*self.T)

    def gso(self, ecap, omega, k):
        r"""
        Gram-Schmidt orthonormalization of the time series.

        ..math::

            e_k(t) = \exp (i \omega_k t)

        with all previous functions.

        Parameters
        ----------
        ecap : array_like
        omega : numeric
            Frequency of current component.
        k : int
            Index of maximum frequency found so far.

        Returns
        -------
        ei : :class:`numpy.ndarray`
            Orthonormalized time series.
        """

        # coefficients
        c_ik = np.zeros(k, dtype=np.complex64)

        u_n = np.exp(1j*omega*self.t)

        # first find the k complex constants cik(k,ndata):
        for j in range(k):
            c_ik[j] = self.hanning_product(u_n, ecap[j])

        # Now construct the orthogonal vector
        e_i = u_n - np.sum(c_ik[:,np.newaxis]*ecap[:k], axis=0)

        # Now normalize this vector
        prod = self.hanning_product(e_i, e_i)

        norm = 1. / np.sqrt(prod)
        if prod == 0.:
            norm = 0. + 0j

        return e_i*norm

    def find_fundamental_frequencies(self, fs, min_freq=1E-6, min_freq_diff=1E-6,
                                     **frecoder_kwargs):
        """
        Solve for the fundamental frequencies of each specified time series,
        `fs`. This is most commonly a 2D array, tuple, or iterable of individual
        complex time series. Any extra keyword arguments are passed to
        `NAFF.frecoder()`.

        Parameters
        ----------
        fs : array_like, iterable
            The time series. If an array-like object, should be 2D with length
            along axis=0 equal to the number of time series.
        min_freq : numeric (optional)
            The minimum (absolute) frequency value to consider non-zero.
        min_freq_diff : numeric (optional)
            The minimum (absolute) frequency difference to distinguish two frequencies.
        **frecoder_kwargs
            Any extra keyword arguments are passed to `NAFF.frecoder()`.

        Returns
        -------

        """

        # containers
        freqs = []
        As = []
        amps = []
        phis = []
        component_ix = []

        nfreqstotal = 0
        ndim = len(fs)

        for i in range(ndim):
            omega,A,phi = self.frecoder(fs[i][:self.n], **frecoder_kwargs)
            freqs.append(omega)  # angular frequencies
            As.append(A*np.exp(1j*phi))  # complex amplitudes
            amps.append(A)  # abs amplitude
            phis.append(phi)  # phase angle
            component_ix.append(np.zeros_like(omega) + i)  # index of the component
            nfreqstotal += len(omega)

        d = np.zeros(nfreqstotal, dtype=zip(('freq','A','|A|','phi','idx'),
                                            ('f8','c8','f8','f8',np.int)))
        d['freq'] = np.concatenate(freqs)
        d['A'] = np.concatenate(As)
        d['|A|'] = np.concatenate(amps)
        d['phi'] = np.concatenate(phis)
        d['idx'] = np.concatenate(component_ix).astype(int)

        # sort terms by amplitude
        d = d[d['|A|'].argsort()[::-1]]  # reverse argsort for descending

        # container arrays for return
        fund_freqs = np.zeros(ndim)
        ffreq_ixes = np.zeros(ndim, dtype=int)
        comp_ixes = np.zeros(ndim, dtype=int)

        # first frequency is largest amplitude, nonzero freq.
        ixes = np.where(np.abs(d['freq']) > min_freq)[0]
        fund_freqs[0] = d[ixes[0]]['freq']
        ffreq_ixes[0] = ixes[0]
        comp_ixes[0] = d[ixes[0]]['idx']

        if ndim == 1:
            return fund_freqs, d, ffreq_ixes

        # choose the next nontrivially related frequency in a different component
        #   as the 2nd fundamental frequency
        abs_freq1 = np.abs(fund_freqs[0])
        ixes = np.where((np.abs(d['freq']) > min_freq) &
                        (d['idx'] != d[ffreq_ixes[0]]['idx']) &  # different component index
                        (np.abs(abs_freq1 - np.abs(d['freq'])) > min_freq_diff))[0]
        fund_freqs[1] = d[ixes[0]]['freq']
        ffreq_ixes[1] = ixes[0]
        comp_ixes[1] = d[ixes[0]]['idx']

        if ndim == 2:
            return fund_freqs[comp_ixes.argsort()], d, ffreq_ixes[comp_ixes.argsort()]

        # third frequency is the largest amplitude frequency in the remaining component dimension
        abs_freq2 = np.abs(fund_freqs[1])
        ixes = np.where((np.abs(d['freq']) > min_freq) &
                        (d['idx'] != d[ffreq_ixes[0]]['idx']) &  # different component index
                        (d['idx'] != d[ffreq_ixes[1]]['idx']) &  # different component index
                        (np.abs(abs_freq1 - np.abs(d['freq'])) > min_freq_diff) &
                        (np.abs(abs_freq2 - np.abs(d['freq'])) > min_freq_diff))[0]

        if len(ixes) == 0:
            # may be a planar orbit
            logger.warning("May be a planar orbit")
            fund_freqs[comp_ixes.argsort()], d, ffreq_ixes[comp_ixes.argsort()]

        fund_freqs[2] = d[ixes[0]]['freq']
        ffreq_ixes[2] = ixes[0]
        comp_ixes[2] = d[ixes[0]]['idx']

        return fund_freqs[comp_ixes.argsort()], d, ffreq_ixes[comp_ixes.argsort()]

    def find_integer_vectors(self, ffreqs, d, imax=15):
        """ TODO """

        ntot = len(d)

        # define meshgrid of integer vectors
        nfreqs = len(ffreqs)
        slc = [slice(-imax,imax+1,None)]*nfreqs
        nvecs = np.vstack(np.vstack(np.mgrid[slc].T))

        # integer vectors
        d_nvec = np.zeros((ntot,nfreqs))
        err = np.zeros(ntot)
        for i in range(ntot):
            this_err = np.abs(d[i]['freq'] - nvecs.dot(ffreqs))
            err[i] = this_err.min()
            d_nvec[i] = nvecs[this_err.argmin()]

        return d_nvec

    def find_actions(self):
        """ Reconstruct approximations to the actions using Percivals equation """
        pass

def orbit_to_freqs(t, w, force_box=False, silently_fail=True, **kwargs):
    """
    Compute the fundamental frequencies of an orbit, ``w``. If not forced, this
    function tries to figure out whether the input orbit is a tube or box orbit and
    then uses the appropriate set of coordinates (Poincaré polar coordinates for tube,
    ordinary Cartesian for box). Any extra keyword arguments (``kwargs``) are passed
    to `NAFF.find_fundamental_frequencies`.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        The orbit to analyze. Should have shape (len(t),6).
    force_box : bool (optional)
        Force the routine to assume the orbit is a box orbit. Default is ``False``.
    silently_fail : bool (optional)
        Return NaN's and None's if NAFF fails, rather than raising an exception.
    **kwargs
        Any extra keyword arguments are passed to `NAFF.find_fundamental_frequencies`.

    """

    if w.ndim == 3:
        # remove extra length-1 dimension (assumed to be axis=1)
        w = w[:,0]

    # now get other frequencies
    if force_box:
        is_tube = False
    else:
        circ = classify_orbit(w)
        is_tube = np.any(circ)

    naff = NAFF(t)

    d = None
    ixes = None
    if is_tube:
        # need to flip coordinates until circulation is around z axis
        new_ws = align_circulation_with_z(w, circ)

        fs = poincare_polar(new_ws)
        if silently_fail:
            try:
                logger.info('Solving for Rφz frequencies...')
                fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)
            except:
                fRphiz = np.ones(3)*np.nan
        else:
            logger.info('Solving for Rφz frequencies...')
            fRphiz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)

        freqs = fRphiz

    else:
        # first get x,y,z frequencies
        logger.info('Solving for XYZ frequencies...')
        fs = [(w[:,j] + 1j*w[:,j+3]) for j in range(3)]

        if silently_fail:
            try:
                fxyz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)
            except:
                fxyz = np.ones(3)*np.nan
        else:
            fxyz,d,ixes = naff.find_fundamental_frequencies(fs, **kwargs)

        freqs = fxyz

    return freqs, d, ixes, is_tube
