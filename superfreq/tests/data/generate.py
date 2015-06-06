# coding: utf-8

""" Generate test data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import h5py
import matplotlib.pyplot as pl
import numpy as np
import gary.potential as gp
from gary.integrate import DOPRI853Integrator
from gary.units import galactic

def main(norbits=100, seed=42):
    np.random.seed(seed)
    cache_path = os.path.split(os.path.abspath(__file__))[0]
    cache_file = os.path.join(cache_path, "isochrone_orbits.h5")

    # integration parameters
    nsteps_per_period = 512
    nperiods = 100
    nsteps = nperiods * nsteps_per_period

    # ---------------------------------------------------------------
    # first data set are orbits integrated in an Isochrone potential

    # potential parameters
    m = 1E11
    b = 5.
    pot = gp.IsochronePotential(m=m, b=b, units=galactic)

    # distance
    r = 10. # arbitrary...

    # velocity magnitude
    menc = pot.mass_enclosed([r,0.,0.])
    vc = np.sqrt(pot.G * menc / r)
    vmag = np.random.normal(vc, vc*0.05, size=norbits)

    # for position
    phi = np.random.uniform(0, 2*np.pi, size=norbits)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.zeros_like(x)
    vx = -vmag * np.sin(phi)
    vy = vmag * np.cos(phi)
    vz = np.zeros_like(x)
    w0 = np.vstack((x,y,z,vx,vy,vz)).T

    # compute true actions, true frequencies
    act,ang,frq = pot.action_angle(w0[:,:3], w0[:,3:])
    # reshape down to 2d
    act = act[:,:2]
    ang = ang[:,:2]
    frq = frq[:,:2]
    true_periods = (2*np.pi / frq).max(axis=-1)

    # integrate them orbits
    dt = np.median(true_periods / nsteps_per_period)
    t,w = pot.integrate_orbit(w0, dt=dt, nsteps=nsteps,
                              Integrator=DOPRI853Integrator)
    w = w[...,[0,1,3,4]]

    # write to file
    f = h5py.File(cache_file, "w")

    truth = f.create_group("truth")
    truth.create_dataset("actions", act.shape, dtype='f8', data=act)
    truth.create_dataset("angles", ang.shape, dtype='f8', data=ang)
    truth.create_dataset("freqs", frq.shape, dtype='f8', data=frq)

    orb = f.create_group("orbits")
    orb.create_dataset("t", t.shape, dtype='f8', data=t)
    orb.create_dataset("w", w.shape, dtype='f8', data=w)

    f.flush()
    f.close()

if __name__ == "__main__":
    main()
