# coding: utf-8

# Third-party
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

# Project
from ..core import orbit_to_fs


def test_orbit_to_fs():
    potential = gp.NFWPotential(m=1E12, r_s=15, b=0.9, c=0.8, units=galactic)
    w0 = gd.CartesianPhaseSpacePosition(pos=[1.,0,0]*u.kpc,
                                        vel=[30.,150,71]*u.km/u.s)
    P = 40*u.Myr
    orbit = potential.integrate_orbit(w0, dt=P/128, t1=0*u.Myr, t2=P*128,
                                      Integrator=gi.DOPRI853Integrator)

    fs = orbit_to_fs(orbit, galactic)
