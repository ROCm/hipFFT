.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation

.. _hipfft-overview:

********************************************************************
hipFFT Overview
********************************************************************

hipFFT is a GPU FFT marshalling library that supports
either `rocFFT`_ or NVIDIA `cuFFT`_ as the backend.

hipFFT exports an interface that does not require the client to
change, regardless of the chosen backend.  It sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.

The basic usage pattern is:

* Create a transform plan (once)
* Perform (many) transforms using the plan
* Destroy the plan when finished

.. _rocFFT: https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html
.. _cuFFT: https://developer.nvidia.com/cufft
