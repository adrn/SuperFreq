# coding: utf-8

from __future__ import division, print_function

""" Note: I got stuck because of the same issue found in the regular code. Using
    the full complex magnitude of phi(w) does better for complex orbits, but for
    a simple case where I construct a time series by hand (with amplitudes and
    frequencies), phi(w) looks like somone took an abs() to the function. e.g.,
    it looks double humped / crosses zero.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
cimport numpy as np

__all__ = ['naff_frequency']

cdef extern from "math.h":
    double sqrt(double)
    double cos(double)
    double sin(double)

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
        zreal[i] = chi[i] * (Re_f[i]*cos(w*tz[i]) + Im_f[i]*sin(w*tz[i]))

        # imag. part of integrand of Eq. 12
        zimag[i] = chi[i] * (Im_f[i]*cos(w*tz[i]) - Re_f[i]*sin(w*tz[i]))

    Re_ans = _simpson(&zreal[0], dtz, ntimes)
    Im_ans = _simpson(&zimag[0], dtz, ntimes)

    ans = sqrt(Re_ans*Re_ans + Im_ans*Im_ans)

    return -(ans*signx) / (2.*T)

cpdef double py_phi_w(double w):
    return phi_w(w)

cpdef double naff_frequency(double omega0, double[::1] _tz, double[::1] _chi,
                            double[::1] _Re_f, double[::1] _Im_f, double _T):
    global ntimes, omin, omax, dtz, T, signx
    global chi, tz, Re_f, Im_f, zreal, zimag

    # local variables
    ntimes = _tz.size
    cdef:
        double xmin = 0.
        double odiff
        double[::1] _zreal = np.zeros(ntimes)
        double[::1] _zimag = np.zeros(ntimes)

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
    omin = omega0 - np.pi/T
    omax = omega0 + np.pi/T
    odiff = omax - omin

    signx = 1.
    local_min(0., 1., 1E-8, 1E-8, phi_w, &xmin)
    if np.allclose(xmin, 0., atol=1E-3) or np.allclose(xmin, 1., atol=1E-3):
        signx = -1.
        local_min(0., 1., 1E-8, 1E-8, phi_w, &xmin)

    # still hits edge
    if np.allclose(xmin, 0., atol=1E-3) or np.allclose(xmin, 1., atol=1E-3):
        raise RuntimeError("Frequency optimizer hit bound.")

    xmin = xmin*(omax - omin) + omin
    return xmin



    w = np.linspace(0, 1, 150)
    phi_vals = np.array([phi_w(ww) for ww in w])

    import time

    import scipy.optimize as so
    t0 = time.time()
    res = so.fmin_slsqp(py_phi_w, x0=0.5, acc=1E-9,
                        bounds=[(0,1)], disp=0, iter=100,
                        full_output=True)
    print("scipy {0:.2f}".format(time.time() - t0))
    scipy_xmin,fx,its,imode,smode = res
    scipy_xmin = scipy_xmin*(omax - omin) + omin

    t0 = time.time()
    local_min(0., 1., 1E-8, 1E-8, phi_w, &xmin)
    xmin = xmin*(omax - omin) + omin
    print("cython {0:.2f}".format(time.time() - t0))
    xmin = xmin*(omax - omin) + omin
    print(scipy_xmin - xmin)
    return 0.

    import matplotlib.pyplot as plt
    plt.plot(w*(omax - omin) + omin, phi_vals)
    plt.axvline(true_omega, color='g')
    plt.axvline(xmin, color='r', linestyle='dashed')
    plt.axvline(scipy_xmin, color='b', linestyle='dashed')
    plt.show()

