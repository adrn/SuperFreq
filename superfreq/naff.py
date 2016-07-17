# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import time
import logging
logger = logging.getLogger(__name__)

# Third-party
import numpy as np
from numpy.fft import fft, fftfreq

# Project
from .core import check_for_primes
from ._naff import naff_frequency
from .simpsgauss import simpson

__all__ = ['SuperFreq', 'find_frequencies',
           'find_integer_vectors', 'closest_resonance',
           'compute_actions']

def hamming(t_T, p):
    return 2.**p * (np.math.factorial(p))**2. / np.math.factorial(2*p) * (1. + np.cos(np.pi*t_T))**p

class SuperFreq(object):
    """
    Implementation of the Numerical Analysis of Fundamental Frequencies
    method of Laskar, later modified by Valluri and Merritt (see references below),
    with some slight modifications.

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
    p : int (optional)
        Coefficient for Hamming filter -- default p=1 which is a Hann filter,
        used by Laskar and Valluri/Merritt.
    keep_calm : bool (optional)
        If something fails when solving for the frequency of a given component,
        ``keep_calm`` determines whether to throw a RuntimeError or exit gracefully.
        If set to ``True``, will exit quietly and carry on to the next component. If
        ``False``, will die if any frequency determination fails.

    """

    def __init__(self, t, p=1, keep_calm=False):

        n = len(t)
        self.n = check_for_primes(n)

        if self.n != len(t):
            logger.debug("Truncating time series to length={0} to avoid large prime divisors."
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

        # pre-compute values of Hamming filter for this window
        # see, e.g., C. Hunter (2001) for a description of why you might want
        # p not equal to 1
        self.chi = hamming(self.tz/self.T, p)

        # when solving for frequencies and removing components from the time series,
        #   if something fails for a given component and keep_calm is set to True,
        #   SuperFreq will exit gracefully instead of throwing a RuntimeError
        self.keep_calm = keep_calm

    def frequency(self, f, omega0=None):
        """
        Find the most significant frequency of a (complex) time series, :math:`f(t)`,
        by Fourier transforming the function convolved with a Hanning filter and
        picking the most significant peak. This assumes the time series, `f`,
        is aligned with / given at the times specified when constructing this
        object. An internal function.

        Parameters
        ----------
        f : array_like
            Complex time-series, e.g., :math:`x(t) + i \, v_x(t)`.
        omega0 : numeric (optional)
            Force finding the peak around the input freuency.

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
        t1 = time.time()
        fff = fft(f) / np.sqrt(self.n)  # NUMPY FFT
        logger.log(0, "Took {} seconds to FFT.".format(time.time()-t1))

        # frequencies
        omegas = 2*np.pi*fftfreq(f.size, self.dt)

        if omega0 is None:
            # omega_max_ix is the initial guess / centering frequency for optimization
            #   against the Hanning-convolved Fourier spectrum
            abs_fff = np.abs(fff)
            omega_max_ix = abs_fff.argmax()
            if np.allclose(abs_fff[omega_max_ix], 0):
                # return early -- "this may be an axial or planar orbit"
                logger.debug("Returning early - may be an axial or planar orbit?")
                return 0.

            # frequency associated with the peak index
            omega0 = omegas[omega_max_ix]

        # real and complex part of input time series
        Re_f = f.real.copy()
        Im_f = f.imag.copy()

        # for debugging -- plot FFT
        # try:
        #     freq = naff_frequency(omega0, self.tz, self.chi, Re_f, Im_f, self.T)
        # except RuntimeError:
        #     import matplotlib.pyplot as plt
        #     plt.clf()
        #     plt.plot(omegas, np.abs(fff), marker=None)
        #     plt.xscale('symlog')
        #     plt.axvline(omega0)
        #     plt.show()
        #     raise
        freq = naff_frequency(omega0, self.tz, self.chi, Re_f, Im_f, self.T)

        return freq

    def frecoder(self, f, nintvec=12, break_condition=1E-7):
        """
        For a given number of iterations, or until the break condition is met:
        solve for strongest frequency of the input time series, subtract
        it from the time series, and iterate.

        Parameters
        ----------
        f : array_like
            Complex time-series, e.g., :math:`x(t) + i \, v_x(t)`.
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
        logger.log(5, "-"*50)
        logger.log(5, "k    ωk    Ak    φk(deg)    ak")
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

            logger.log(5, "{}  {:.6f}  {:.6f}  {:.2f}  {:.6f}"
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
        Solve for the fundamental frequencies of each specified time series.

        This is most commonly a 2D array, a tuple, or iterable of individual
        complex time series. For example, if your orbit is 2D, you might pass in
        a tuple with :math:`x +i \, v_x` as the 0th element and :math:`y +i \, v_y`
        as the 1st element.

        Any extra keyword arguments are passed to `SuperFreq.frecoder()`.

        Parameters
        ----------
        fs : array_like, iterable
            The iterable of (complex) time series. If an array-like object, should
            be 2D with length along axis=0 equal to the number of time series. See
            description above.
        min_freq : numeric (optional)
            The minimum (absolute) frequency value to consider a non-zero
            frequency component.
        min_freq_diff : numeric (optional)
            The minimum (absolute) frequency difference to distinguish two
            frequencies.
        **frecoder_kwargs
            Any extra keyword arguments are passed to `SuperFreq.frecoder()`.

        Returns
        -------
        freqs : :class:`numpy.ndarray`
            The fundamental frequencies of the orbit. This will have the same
            number of elements as the dimensionality of the orbit.
        table : :class:`numpy.ndarray`
            The full table of frequency modes, amplitudes, and phases for all
            components detected in the FFT.
        freq_ixes : :class:`numpy.ndarray`
            The indices of the rows of the table that correspond to the modes
            identified as the fundamental frequencies.
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

        if len(ixes) == 0 and self.keep_calm:
            # may be a planar orbit
            logger.warning("May be a planar orbit")
            return fund_freqs[comp_ixes.argsort()], d, ffreq_ixes[comp_ixes.argsort()]

        fund_freqs[2] = d[ixes[0]]['freq']
        ffreq_ixes[2] = ixes[0]
        comp_ixes[2] = d[ixes[0]]['idx']

        return fund_freqs[comp_ixes.argsort()], d, ffreq_ixes[comp_ixes.argsort()]

def find_integer_vectors(freqs, table, max_int=12):
    r"""
    Given the fundamental frequencies and table of all frequency
    components, determine how each frequency component is related
    to the fundamental frequencies (e.g., determine the integer
    vector for each frequency such that :math:`n\cdot \Omega \approx 0.`).
    These are the non-zero Fourier modes.

    Parameters
    ----------
    freqs : array_like
        The fundamental frequencies.
    table : structured array
        The full table of frequency modes, amplitudes, and phases for all
        components detected in the FFT.
    max_int : int (optional)
        The integer vectors considered will go from ``-max_int`` to ``max_int``
        in each dimension.

    Returns
    -------
    nvec : :class:`numpy.ndarray`
        An array of integer vectors that correspond to the frequency components.

    """

    # make sure the fundamental frequencies are a numpy array
    freqs = np.array(freqs)

    ncomponents = len(table)

    # define meshgrid of integer vectors
    nfreqs = len(freqs)
    slc = [slice(-max_int,max_int+1,None)]*nfreqs
    nvecs = np.vstack(np.vstack(np.mgrid[slc].T))

    # integer vectors
    d_nvec = np.zeros((ncomponents,nfreqs)).astype(int)
    for i in range(ncomponents):
        errs = np.abs(nvecs.dot(freqs) - table[i]['freq'])
        d_nvec[i] = nvecs[errs.argmin()]

    return d_nvec

def closest_resonance(freqs, max_int=12):
    r"""
    Find the closest resonant vector for the given set of fundamental
    frequencies.

    Parameters
    ----------
    freqs : array_like
        The fundamental frequencies.
    max_int : int (optional)
        The integer vectors considered will go from ``-max_int`` to ``max_int``
        in each dimension.

    Returns
    -------
    intvec : :class:`numpy.ndarray`
        The integer vector of the closest resonance.
    dist : float
        The distance to the closest resonance, e.g., if :math:`\boldsymbol{n}`
        is the resonant integer vector, this is just
        :math:`\boldsymbol{n} \cdot \boldsymbol{\Omega}`.

    """

    # make sure the fundamental frequencies are a numpy array
    freqs = np.array(freqs)

    # define meshgrid of integer vectors
    nfreqs = len(freqs)
    slc = [slice(-max_int,max_int+1,None)]*nfreqs
    nvecs = np.vstack(np.vstack(np.mgrid[slc].T))

    ndf = nvecs.dot(freqs)
    min_ix = ndf.argmin()

    return nvecs[min_ix], ndf[min_ix]

def find_frequencies(t, w, force_box=False, silently_fail=True, **kwargs):
    """
    Compute the fundamental frequencies of an orbit, ``w``. If not forced, this
    function tries to figure out whether the input orbit is a tube or box orbit and
    then uses the appropriate set of coordinates (Poincaré polar coordinates for tube,
    ordinary Cartesian for box). Any extra keyword arguments (``kwargs``) are passed
    to `SuperFreq.find_fundamental_frequencies`.

    Requires Gala.

    Parameters
    ----------
    t : array_like
        Array of times.
    w : array_like
        The orbit to analyze. Should have shape (len(t),6).
    force_box : bool (optional)
        Force the routine to assume the orbit is a box orbit. Default is ``False``.
    silently_fail : bool (optional)
        Return NaN's and None's if SuperFreq fails, rather than raising an exception.
    **kwargs
        Any extra keyword arguments are passed to `SuperFreq.find_fundamental_frequencies`.

    """

    from gala.dynamics import classify_orbit, align_circulation_with_z
    from gala.coordinates import cartesian_to_poincare_polar

    if w.ndim == 3:
        # remove extra length-1 dimension (assumed to be axis=1)
        w = w[:,0]

    # now get other frequencies
    if force_box:
        is_tube = False
    else:
        circ = classify_orbit(w)
        is_tube = np.any(circ)

    naff = SuperFreq(t)

    d = None
    ixes = None
    if is_tube:
        # need to flip coordinates until circulation is around z axis
        new_ws = align_circulation_with_z(w, circ)
        new_ws = cartesian_to_poincare_polar(new_ws)
        fs = [(new_ws[:,j] + 1j*new_ws[:,j+3]) for j in range(3)]

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

def compute_actions(freqs, table, max_int=12):
    """
    Reconstruct approximations to the actions using Percival's equation.

    Approximate the values of the actions using a Fourier decomposition.
    You must pass in the frequencies and frequency table determined from
    and orbit in Cartesian coordinates.

    For example, see:
    - `Valluri & Merritt (1999) <http://arxiv.org/abs/astro-ph/9906176>`_
    - `Percival (1974) <http://iopscience.iop.org/0301-0015/7/7/005>`_

    Parameters
    ----------
    freqs : array_like
        The fundamental frequencies.
    table : structured array
        The full table of frequency modes, amplitudes, and phases for all
        components detected in the FFT.
    max_int : int (optional)
        The integer vectors considered will go from ``-max_int`` to ``max_int``
        in each dimension.

    Returns
    -------
    actions : :class:`numpy.ndarray`
        Numerical estimates of the orbital actions.

    """
    ndim = len(freqs)

    # get integer vectors for each component
    nvecs = find_integer_vectors(freqs, table, max_int=max_int)

    # container to store |X_k|^2
    amp2 = np.zeros([2*max_int+2]*ndim)
    for nvec,row in zip(nvecs, table):
        slc = [slice(x+max_int,x+max_int+1,None) for x in nvec]
        amp2[slc] += row['A'].real**2

    Js = np.zeros(ndim)
    for nvec in nvecs:
        slc = [slice(x+max_int,x+max_int+1,None) for x in nvec]
        Js += nvec * nvec.dot(freqs) * float(amp2[slc])

    return Js
