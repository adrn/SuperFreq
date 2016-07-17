###############
Getting started
###############

SuperFreq operates on time series, e.g., orbits. This package does not provide
any functionality for integrating orbits or transforming between coordinate
representations (for that, you might want to install `astropy
<https://github.com/astropy/astropy>`__ and `gala
<https://github.com/adrn/gala>`__).

Running SuperFreq on a pre-generated orbit
==========================================

The first step will be to generate or read in the orbit data. For this example,
let's imagine the orbit is saved as two Numpy binary files (one for the time
array, one for the orbit itself):

.. code-block:: python

   import numpy as np
   t = np.load("time.npy")
   w = np.load("orbit.npy")
   ntimes, ndim = w.shape

We'll assume the time array, ``t``, is a 1D Numpy array, and the orbit array,
``w``, is a 2D array with shape equal to ``(ntimes, ndim)`` where ``ndim`` is
the phase-space dimensionality. We next create a :class:`~superfreq.SuperFreq`
object by passing in the array of times to the initializer:

.. code-block:: python

   from superfreq import SuperFreq
   sf = SuperFreq(t)

This is the object that we will use to do the frequency analysis. We next have
to define the complex time series from which we would like to derive the
fundamental frequencies. As demonstrated in many papers by J. Laskar and M.
Valluri / D. Merritt, passing in complex combinations of the phase-space
coordinates seems to lead to more accurate recovery of the frequencies (e.g.,
:math:`x(t) + i \, v_x(t)` and etc.). We therefore define three such complex
arrays from the orbit data:

.. code-block:: python

   fs = [(w[:,i] * 1j*w[:,i+ndim//2]) for i in range(ndim//2)]

We can now run the frequency solver on this list of arrays:

.. code-block:: python

   freqs, tbl, ix = sf.find_fundamental_frequencies(fs)

In the returned variables above, ``freqs`` will contain the fundamental
frequencies.
