.. toctree::
   :maxdepth: 4
   :caption: Contents:

======
hipFFT
======

hipFFT is a GPU FFT marshalling library. Currently, hipFFT supports
either `rocFFT`_ or `cuFFT`_ as backends.

hipFFT exports an interface that does not require the client to
change, regardless of the chosen backend.  It sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.

The basic usage pattern is:

* create a transform plan (once)
* perform (many) transforms using the plan
* destroy the plan

.. _rocFFT: https://rocm.docs.amd.com/projects/rocFFT/en/latest/index.html
.. _cuFFT: https://developer.nvidia.com/cufft
