# coding: utf-8

# Third-party
from astropy.table import Table
import numpy as np

__all__ = ['check_for_primes', 'orbit_to_fs']


def check_for_primes(n, max_prime=41):
    """
    Given an integer, ``n``, ensure that it doest not have large prime
    divisors, which can wreak havok for FFT's. If needed, will decrease
    the number.

    Parameters
    ----------
    n : int
        Integer number to test.

    Returns
    -------
    n2 : int
        Integer combed for large prime divisors.
    """

    m = n
    f = 2
    while (f**2 <= m):
        if m % f == 0:
            m /= f
        else:
            f += 1

    if m >= max_prime and n >= max_prime:
        n -= 1
        n = check_for_primes(n)

    return n


def orbit_to_fs(orbit, units, style='laskar'):
    r"""
    Convert the orbit (position and velocity time series) into complex
    time series to be analyzed by `SuperFreq`.

    For `style=='laskar'`, assumes the standard complex time series

    .. math::

        f_i = q_i + i\,p_i

    where :math:`q_i,p_i` are the coordinate and conjugate momenta
    (e.g., velocity for m=1).

    Parameters
    ----------
    orbit : :class:`gala.dynamics.CartesianOrbit`
        The input orbit.
    units : :class:`gala.units.UnitSystem`
        The unit system.
    style : str (optional)
        Currently only supports `style = 'laskar'`.

    Returns
    -------
    fs : tuple

    """

    style = str(style).lower().strip()

    if style != 'laskar':
        raise ValueError("Currently only supports style = 'laskar'")

    q = np.squeeze(orbit.pos.xyz.decompose(units).value)
    p = np.squeeze(orbit.vel.d_xyz.decompose(units).value)

    if q.ndim > 2:
        raise ValueError("This function only supports converting single "
                         "orbits, not a collection of orbits.")
    elif q.ndim < 2:
        raise ValueError("The orbit must contain more than one timestep.")

    fs = [q[i] + 1j*p[i] for i in range(q.shape[0])]

    return fs


class SuperFreqResult:

    def __init__(self, fund_freqs, freq_mode_table, fund_freqs_idx):
        self.fund_freqs = fund_freqs
        self.fund_freqs_idx = fund_freqs_idx
        self.freq_mode_table = Table(freq_mode_table)

        # derived / computed
        self.fund_freq_amps = np.asarray(self.freq_mode_table[self.fund_freqs_idx]['|A|'])

    def model_f(self, t, component_idx):
        """
        Parameters
        ----------
        t : array_like
            Array of times to evaluate the fourier sum model time series.
        component_idx : int
            The component to create the model time series for (e.g., the
            index of `fs`).

        Returns
        -------
        model_f : :class:`numpy.ndarray`
        """
        t = np.asarray(t)
        tbl = self.freq_mode_table[self.freq_mode_table['idx'] == component_idx]
        X = tbl['A'][None] * np.exp(1j*tbl['freq'][None] * t[:,None])
        model_f = X.sum(axis=1)
        return model_f
