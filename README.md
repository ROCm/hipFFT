# hipFFT
hipFFT is a FFT marshalling library, with multiple supported backends.  It sits between the application and a 'worker' FFT library, marshalling inputs into the backend library and marshalling results back to the application.  hipFFT exports an interface that does not require the client to change, regardless of the chosen backend.  Currently, hipFFT supports [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT) and [cuFFT](https://developer.nvidia.com/cufft) as backends.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories) or by clicking the github releases tab and manually downloading, which could be newer.  Release notes are available for each release on the releases tab.
* `sudo apt update && sudo apt install hipfft`

## Quickstart hipFFT build

#### Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script `install.sh` to build and install hipFFT on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.  A few commands in the script need sudo access, so it may prompt you for a password.
*  `./install -h` -- shows help
*  `./install -id` -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

## Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the [hipfft build wiki](https://github.com/ROCmSoftwarePlatform/hipFFT/wiki/Build) has helpful information on how to configure cmake and manually build.

### Functions supported
A list of [exported functions](https://github.com/ROCmSoftwarePlatform/hipFFT/wiki/Exported-functions) from hipfft can be found on the wiki
