# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
from distutils.core import setup
from distutils.extension import Extension

# Third-party
import numpy as np
from Cython.Distutils import build_ext
from Cython.Build import cythonize

# Get numpy path
numpy_base_path = os.path.split(np.__file__)[0]
numpy_incl_path = os.path.join(numpy_base_path, "core", "include")

extensions = []
extensions.append(Extension("superfreq",
                            ["superfreq/*.pyx",
                             "superfreq/brent.c",
                             "superfreq/simpson.c"],
                            include_dirs=[numpy_incl_path]))

setup(
    name="superfreq",
    version="0.1",
    author="Adrian M. Price-Whelan",
    author_email="adrn@astro.columbia.edu",
    license="MIT",
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    packages=["superfreq"],
    package_data={'superfreq': ['*.pxd','*.c']},
)
