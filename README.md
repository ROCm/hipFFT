# hipFFT

hipFFT is an FFT marshalling library. Currently, hipFFT supports
either [rocFFT] or [cuFFT] as backends.

hipFFT exports an interface that does not require the client to
change, regardless of the chosen backend.  It sits between the
application and the backend FFT library, marshalling inputs into the
backend and results back to the application.

[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
[cuFFT]: https://developer.nvidia.com/cufft

## Installing pre-built packages

Download pre-built packages either from [ROCm's package servers].

* On Ubuntu: `sudo apt update && sudo apt install hipfft`

[ROCm's package servers]: https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html

## Transitioning from rocFFT

If you are transitioning from the hipFFT version included in rocFFT to
this standalone hipFFT version; please modify your build following
this example:

* previously: `hipcc hipfft_1d_z2z.cpp -L/opt/rocm/lib -lrocfft`
* during transition: `hipcc -I/opt/rocm/hipfft/include hipfft_1d_z2z.cpp -L/opt/rocm/lib -lhipfft -lrocfft`

## Building from source

### Library build dependencies

To build the hipFFT library:
* hipFFT depends on [rocFFT] on AMD platforms;
* hipFFT depends on [cuFFT] on Nvidia platforms.

### Client build dependencies

* The clients (samples, tests etc) included with the hipFFT source
  depend on FFTW, gtest, and boost program-options.

* The rider and test clients also require the rocFFT source tree to
  build:

    git submodule update --init

### Building hipFFT

To show all build options:

    mkdir build && cd build
    cmake -LH ..

Here are some CMake build examples:

| Hardware target | Case                                                                             | Build command line                                                                                       |
| ---             | ---                                                                              | ---                                                                                                      |
| AMD GPU         | Build a project using HIP language APIs + hipFFT with standard host compiler     | cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -L ..                       |
| AMD GPU         | Build a project using HIP language APIs + hipFFT + device kernels with HIP-clang | cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L ..                     |
| NVIDIA GPU      | Build a project using HIP language APIs + hipFFT with standard host compiler     | cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_LIB=CUDA -L .. |
| NVIDIA GPU      | Build a project using HIP language APIs + hipFFT + device kernels with HIP-nvcc  | HIP_PLATFORM=nvidia cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIENTS=ON -L .. |

Note that the option -DBUILD_CLIENTS=ON is only allowed for the hipcc compiler.

## Quick CUDA porting guide

If you have existing CUDA code and want to transition to HIP:
* [HIPIFY] your code and fix all unsupported CUDA features or user-defined macros.
* Build with HIP-nvcc to run on an Nvidia device.
* Build with HIP-clang to run on an AMD device.

More information about porting to HIP is available on the [HIP porting guide].

[HIPIFY]: https://github.com/ROCm-Developer-Tools/HIPIFY
[HIP porting guide]: https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html
