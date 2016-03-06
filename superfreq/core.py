# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
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
    orbit : :class:`gary.dynamics.CartesianOrbit`
        The input orbit.
    units : :class:`gary.units.UnitSystem`
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

    q = np.squeeze(orbit.pos.decompose(units).value)
    p = np.squeeze(orbit.vel.decompose(units).value)

    if q.ndim > 2:
        raise ValueError("This function only supports converting single "
                         "orbits, not a collection of orbits.")
    elif q.ndim < 2:
        raise ValueError("The orbit must contain more than one timestep.")

    fs = [q[i] + 1j*p[i] for i in range(q.shape[0])]

    return fs

# TODO:
class SuperFreqResult(object):
    pass
