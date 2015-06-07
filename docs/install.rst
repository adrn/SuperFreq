############
Installation
############

Requirements
============

This packages has the following strict requirements:

- `Python <http://www.python.org/>`_ 2.7, 3.3, 3.4

- `Numpy <http://www.numpy.org/>`_ 1.8 or later

- `Cython <http://www.cython.org/>`_: 0.21 or later

You can use pip to install these automatically using the
`pip-requirements.txt <https://github.com/adrn/SuperFreq/blob/master/pip-requirements-txt>`_
file (from the root of the project):

   pip install -r pip-requirements.txt

Installing
==========

Development repository
----------------------

The latest development version of superfreq can be cloned from
`GitHub <https://github.com/>`_ using git::

   git clone git://github.com/adrn/superfreq.git

Building and Installing
-----------------------

To build the project (from the root of the source tree, e.g., inside
the cloned ``superfreq`` directory)::

   python setup.py build

To install the project::

   python setup.py install
