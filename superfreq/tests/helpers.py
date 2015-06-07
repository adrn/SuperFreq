# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

def cartesian_to_poincare(w):
    # assuming z == 0
    R = np.sqrt(w[...,0]**2 + w[...,1]**2)
    phi = np.arctan2(w[...,0], w[...,1])

    vR = (w[...,0]*w[...,0+2] + w[...,1]*w[...,1+2]) / R
    vPhi = w[...,0]*w[...,1+2] - w[...,1]*w[...,0+2]

    # pg. 437, Papaphillipou & Laskar (1996)
    sqrt_2THETA = np.sqrt(np.abs(2*vPhi))
    pp_phi = sqrt_2THETA * np.cos(phi)
    pp_phidot = -sqrt_2THETA * np.sin(phi)

    new_w = np.vstack((R.T, pp_phi.T, vR.T, pp_phidot.T)).T
    return new_w
