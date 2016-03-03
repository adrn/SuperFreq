# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

def cartesian_to_poincare(xy, vxy):
    # assuming z == 0
    R = np.sqrt(xy[0]**2 + xy[1]**2)
    phi = np.arctan2(xy[0], xy[1])

    vR = (xy[0]*vxy[0] + xy[1]*vxy[1]) / R
    vPhi = xy[0]*vxy[1] - xy[1]*vxy[0]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = sqrt_2THETA * np.sin(phi)

    rphi = np.vstack((R, pp_phi))
    vrphi = np.vstack((vR, pp_phidot))
    return rphi, vrphi
