# hipFFT
hipFFT is a FFT marshalling library, with multiple supported backends.  It sits between the application and a 'worker' FFT library, marshalling inputs into the backend library and marshalling results back to the application.  hipFFT exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipFFT supports [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT) and [cuFFT](https://developer.nvidia.com/cufft) as backends.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories) or by clicking the github releases tab and manually downloading, which could be newer.  Release notes are available for each release on the releases tab.
* `sudo apt update && sudo apt install hipfft`

## Quickstart hipFFT build

## Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the [hipfft build wiki](https://github.com/ROCmSoftwarePlatform/hipFFT/wiki/Build) has helpful information on how to configure cmake and manually build.

### Functions supported
A list of [exported functions](https://github.com/ROCmSoftwarePlatform/hipFFT/wiki/Exported-functions) from hipfft can be found on the wiki

