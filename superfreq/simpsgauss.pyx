# coding: utf-8

# Standard library
import os
import sys

# Third-party
import numpy as np

__all__ = ['simpson']

cdef extern from "simpson.h":
    double _simpson (double *y, double dx, int n)

cpdef simpson(double[::1] y, double dx):
    return _simpson(&y[0], dx, y.size)
