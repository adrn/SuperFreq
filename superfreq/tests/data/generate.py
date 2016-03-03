# coding: utf-8

""" Generate test data. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
from astropy.utils.data import _find_pkg_data_path
import h5py
import numpy as np
import gary.dynamics as gd
import gary.potential as gp
from gary.integrate import DOPRI853Integrator
from gary.units import galactic

def get_isochrone_orbits(n_orbits=100, seed=42):
    np.random.seed(seed)

    cache_file = os.path.abspath(_find_pkg_data_path("isochrone_orbits.h5"))

    if os.path.exists(cache_file):
        return cache_file

    # integration parameters
    nperiods = 50
    nsteps_per_period = 512
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
    vmag = np.random.normal(vc-0.01, vc*0.01, size=n_orbits)

    # for position
    phi = np.random.uniform(0, 2*np.pi, size=n_orbits)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.zeros_like(x)
    vx = -vmag * np.sin(phi)
    vy = vmag * np.cos(phi)
    vz = np.zeros_like(x)
    pos = np.vstack((x,y,z))*galactic['length']
    vel = np.vstack((vx,vy,vz))*galactic['length']/galactic['time']

    # compute true actions, true frequencies
    w0 = gd.CartesianPhaseSpacePosition(pos=pos, vel=vel)
    act,ang,frq = pot.action_angle(w0)

    # reshape down to 2d
    act = act[:2]
    ang = ang[:2]
    frq = frq[:2]
    true_periods = (2*np.pi / frq).max(axis=0)

    # write to file
    f = h5py.File(cache_file, "w")

    truth = f.create_group("initial")
    truth.create_dataset("actions", act.shape, dtype='f8', data=act.value)
    truth["actions"].attrs['unit'] = str(act.unit)

    truth.create_dataset("angles", ang.shape, dtype='f8', data=ang.value)
    truth["angles"].attrs['unit'] = str(ang.unit)

    truth.create_dataset("freqs", frq.shape, dtype='f8', data=frq.value)
    truth["freqs"].attrs['unit'] = str(frq.unit)

    # integrate them orbits -- have to do it this way to make sure
    #   dt is right
    all_x = np.zeros((2,nsteps+1,n_orbits))
    all_v = np.zeros((2,nsteps+1,n_orbits))
    all_t = np.zeros((nsteps+1,n_orbits))
    for i,period in enumerate(true_periods):
        print("Orbit {0}".format(i))
        dt = period / nsteps_per_period
        orbit = pot.integrate_orbit(w0[i], dt=dt, nsteps=nsteps,
                                    Integrator=DOPRI853Integrator)
        all_x[...,i] = orbit.pos.decompose(pot.units).value[:2]
        all_v[...,i] = orbit.vel.decompose(pot.units).value[:2]
        all_t[...,i] = orbit.t.decompose(pot.units).value

    orb = f.create_group("orbits")
    orb.create_dataset("t", all_t.shape, dtype='f8', data=all_t)
    orb['t'].attrs['unit'] = str(pot.units['time'])

    orb.create_dataset("x", all_x.shape, dtype='f8', data=all_x)
    orb['x'].attrs['unit'] = str(pot.units['length'])

    orb.create_dataset("v", all_v.shape, dtype='f8', data=all_v)
    orb['v'].attrs['unit'] = str(pot.units['length']/pot.units['time'])

    f.flush()
    f.close()

    return cache_file
