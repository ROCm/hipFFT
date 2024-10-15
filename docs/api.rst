.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation

.. _hipfft-api-usage:

********************************************************************
hipFFT API Usage
********************************************************************

This section describes usage of the hipFFT library API.  The hipFFT
API follows the NVIDIA `cuFFT`_ API.

.. _cuFFT: https://docs.nvidia.com/cuda/cufft/

Types
=====

There are a few data structures that are internal to the library.  The
pointer types to these structures are given below.  The user would
need to use these types to create handles and pass them between
different library functions.

.. doxygendefine:: HIPFFT_FORWARD

.. doxygendefine:: HIPFFT_BACKWARD

.. doxygenenum:: hipfftType_t

.. doxygentypedef:: hipfftHandle

.. doxygenenum:: hipfftResult_t


Simple plans
============

These planning routines allocate a plan for you.  If execution of the
plan requires a work buffer, it will be created (and destroyed)
automatically.

.. doxygenfunction:: hipfftPlan1d

.. doxygenfunction:: hipfftPlan2d

.. doxygenfunction:: hipfftPlan3d


User managed simple plans
-------------------------

These planning routines assume that you have allocated a plan
(`hipfftHandle`) yourself; and that you will manage a work area as
well.

If you want to manage your own work buffer... XXX

.. doxygenfunction:: hipfftCreate

.. doxygenfunction:: hipfftDestroy

.. doxygenfunction:: hipfftSetAutoAllocation

.. doxygenfunction:: hipfftMakePlan1d

.. doxygenfunction:: hipfftMakePlan2d

.. doxygenfunction:: hipfftMakePlan3d


Advanced plans
===================

.. doxygenfunction:: hipfftMakePlanMany
.. doxygenfunction:: hipfftXtMakePlanMany



Estimating work area sizes
==========================

These call return estimates of the work area required to support a
plan generated with the same parameters (either with the simple or
extensible API).  Callers who choose to manage work area allocation
within their application must use this call after plan generation, and
after any hipfftSet*() calls subsequent to plan generation, if those
calls might alter the required work space size.

.. doxygenfunction:: hipfftEstimate1d

.. doxygenfunction:: hipfftEstimate2d

.. doxygenfunction:: hipfftEstimate3d

.. doxygenfunction:: hipfftEstimateMany


Accurate work area sizes
------------------------

After plan generation is complete, an accurate work area size can be
obtained with these routines.

.. doxygenfunction:: hipfftGetSize1d

.. doxygenfunction:: hipfftGetSize2d

.. doxygenfunction:: hipfftGetSize3d

.. doxygenfunction:: hipfftGetSizeMany

.. doxygenfunction:: hipfftXtGetSizeMany
		     

Executing plans
===============

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

.. doxygenfunction:: hipfftXtExec
		     

Callbacks
=========

.. doxygenfunction:: hipfftXtSetCallback
.. doxygenfunction:: hipfftXtClearCallback	     
.. doxygenfunction:: hipfftXtSetCallbackSharedSize

		     
Single-process Multi-GPU Transforms
===================================

hipFFT offers experimental single-process multi-GPU transforms.
		     
Multiple devices can be associated to a :cpp:type:`hipfftHandle` using
:cpp:func:`hipfftXtSetGPUs`.  Once a plan is associated to multiple
GPUs, :cpp:struct:`hipLibXtDesc` is used to pass multiple GPU buffers
to the plan for execution.  These buffers are allocated via
:cpp:func:`hipfftXtMalloc`, and free'd with :cpp:func:`hipfftXtFree`.
The function :cpp:func:`hipfftXtMemcpy` allows one to move data to or
from a :cpp:struct:`hipLibXtDesc` and a contiguous host buffer, or
between two :cpp:struct:`hipLibXtDesc` s.

Execution is performed with the appropriate
:cpp:func:`hipfftXtExecDescriptor`

.. doxygenfunction:: hipfftXtSetGPUs

.. doxygenstruct:: hipXtDesc_t
.. doxygenstruct:: hipLibXtDesc_t

.. doxygenfunction:: hipfftXtMalloc
.. doxygenfunction:: hipfftXtFree
.. doxygenfunction:: hipfftXtMemcpy
		     
.. doxygengroup:: hipfftXtExecDescriptor
