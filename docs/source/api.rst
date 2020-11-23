.. toctree::
   :maxdepth: 4
   :caption: Contents:

=========
API Usage
=========

This section describes usage of the hipFFT library API.  The hipFFT
API follows the `cuFFT`_ API.

.. _cuFFT: https://docs.nvidia.com/cuda/cufft/

Types
-----

There are a few data structures that are internal to the library.  The
pointer types to these structures are given below.  The user would
need to use these types to create handles and pass them between
different library functions.

.. doxygendefine:: HIPFFT_FORWARD

.. doxygendefine:: HIPFFT_BACKWARD

.. doxygenenum:: hipfftType

.. doxygentypedef:: hipfftHandle

.. doxygenenum:: hipfftResult


Simple plans
------------

These planning routines allocate a plan for you.  If execution of the
plan requires a work buffer, it will be created (and destroyed)
automatically.

.. doxygenfunction:: hipfftPlan1d

.. doxygenfunction:: hipfftPlan2d

.. doxygenfunction:: hipfftPlan3d


User managed simple plans
-------------------------

These planning routines assume that you have created a plan
(`hipfftHandle`) yourself.

If you want to manage your own work buffer... XXX

.. doxygenfunction:: hipfftCreate

.. doxygenfunction:: hipfftDestroy

.. doxygenfunction:: hipfftSetAutoAllocation

.. doxygenfunction:: hipfftMakePlan1d

.. doxygenfunction:: hipfftMakePlan2d

.. doxygenfunction:: hipfftMakePlan3d


More advanced plans
-------------------

Executing plans
---------------

Once you have created an FFT plan, you can execute it using one of the
`hipfftExec*` functions.

For real-to-complex transforms, the output buffer XXX

For complex-to-real transforms, the output buffer XXX

.. doxygenfunction:: hipfftExecC2C

.. doxygenfunction:: hipfftExecR2C

.. doxygenfunction:: hipfftExecC2R

.. doxygenfunction:: hipfftExecZ2Z

.. doxygenfunction:: hipfftExecD2Z

.. doxygenfunction:: hipfftExecZ2D
