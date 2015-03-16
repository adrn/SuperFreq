# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np
from astropy import log as logger
import astropy.units as u

# Project
# ...

__all__ = ['naff_frequency']

cdef extern from "math.h":
    double sqrt(double)

cdef extern from "../integrate/1d/simpson.h":
    double _simpson (double *y, double dx, int n)

cdef extern from "brent.h":
    double local_min (double a, double b, double eps, double t,
                      double f(double x), double *x)

# variables needed within phi_w
cdef int ntimes
cdef double omin, omax, dtz, T, signx
cdef:
    double *chi
    double *tz
    double *Re_f
    double *Im_f
    double *zreal
    double *zimag

cdef double phi_w(double w):
    """ φ(ω) from Laskar (1990) or Valluri & Merritt (1998) Eq. 12 """

    cdef int i

    # un-transform the frequency so it spans it's initial domain
    w = w*(omax - omin) + omin

    for i in range(ntimes):
        # real part of integrand of Eq. 12
        zreal[i] = chi[i] * (Re_f[i]*np.cos(w*tz[i]) + Im_f[i]*np.sin(w*tz[i]))

        # imag. part of integrand of Eq. 12
        zimag[i] = chi[i] * (Im_f[i]*np.cos(w*tz[i]) - Re_f[i]*np.sin(w*tz[i]))

    Re_ans = _simpson(&zreal[0], dtz, ntimes)
    Im_ans = _simpson(&zimag[0], dtz, ntimes)

    ans = sqrt(Re_ans*Re_ans + Im_ans*Im_ans)

    return -(ans*signx) / (2.*T)

cpdef double naff_frequency(double[::1] _tz, double[::1] _chi,
                            double[::1] _Re_f, double[::1] _Im_f, double _T):
    global ntimes, omin, omax, dtz, T, signx
    global chi, tz, Re_f, Im_f, zreal, zimag

    # local variables
    ntimes = _tz.size
    cdef double xmin = 0.
    cdef double[::1] _zreal = np.zeros(ntimes)
    cdef double[::1] _zimag = np.zeros(ntimes)

    # test
    cdef double true_omega

    # global variables
    chi = &_chi[0]
    tz = &_tz[0]
    Re_f = &_Re_f[0]
    Im_f = &_Im_f[0]
    zreal = &_zreal[0]
    zimag = &_zimag[0]
    T = _T
    dtz = tz[1] - tz[0]

    # local variables
    true_omega = 0.581
    omin = true_omega - np.pi/T
    omax = true_omega + np.pi/T
    signx = 1.

    local_min(0., 1., 1E-10, 1E-10, phi_w, &xmin)
    xmin = xmin*(omax - omin) + omin
    print(xmin)

