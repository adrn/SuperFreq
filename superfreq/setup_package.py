# Licensed under a 3-clause BSD style license - see PYFITS.rst
from __future__ import absolute_import

from distutils.core import Extension

from astropy_helpers import setup_helpers


def get_extensions():
    cfg_naff = setup_helpers.DistutilsExtensionArgs()
    # 'numpy' will be replaced with the proper path to the numpy includes
    cfg_naff['include_dirs'].append('numpy')
    cfg_naff['include_dirs'].append('cextern')
    cfg_naff['sources'].append('superfreq/_naff.pyx')
    cfg_naff['sources'].append('cextern/brent.c')
    cfg_naff['sources'].append('cextern/simpson.c')
    ext_naff = Extension('superfreq._naff', **cfg_naff)

    cfg_simpsgauss = setup_helpers.DistutilsExtensionArgs()
    cfg_simpsgauss['include_dirs'].append('numpy')
    cfg_simpsgauss['include_dirs'].append('cextern')
    cfg_simpsgauss['sources'].append('superfreq/simpsgauss.pyx')
    cfg_simpsgauss['sources'].append('cextern/simpson.c')
    ext_simpsgauss = Extension('superfreq.simpsgauss', **cfg_simpsgauss)

    return [ext_naff, ext_simpsgauss]


# are the c files provided by some external library?  If so, should add it here
# so that users have the option of using that instead of the builtin one
#def get_external_libraries():
#    return ['something']
