# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## hipFFT 1.0.12 for ROCm 5.6.0

### Added
- Implemented the hipfftXtMakePlanMany, hipfftXtGetSizeMany, hipfftXtExec APIs, to allow requesting half-precision transforms.

### Changed
- Added --precision argument to benchmark/test clients.  --double is still accepted but is deprecated as a method to request a double-precision transform.

## hipFFT 1.0.11 for ROCm 5.5.0

### Fixed

- Fixed old version rocm include/lib folders not removed on upgrade.

## hipFFT 1.0.10 for ROCm 5.4.0

### Added
- Added hipfftExtPlanScaleFactor API to efficiently multiply each output element of a FFT by a given scaling factor.  Result scaling must be supported in the backend FFT library.

### Changed
- When hipFFT is built against the rocFFT backend, rocFFT 1.0.19 or higher is now required.
- Data is initialized directly on GPUs using hipRAND.
- Updated build files to use standard C++17.

## hipFFT 1.0.9 for ROCm 5.3.0

### Changed

- Clean up build warnings.
- GNUInstall Dir enhancements.
- Requires gtest 1.11.

## hipFFT 1.0.8 for ROCm 5.2.0

### Added
- Added File/Folder Reorg Changes with backward compatibility support using ROCM-CMAKE wrapper functions.
- Packages for test and benchmark executables on all supported OSes using CPack.
- Implemented hipfftMakePlanMany64 and hipfftGetSizeMany64.

## hipFFT 1.0.7 for ROCm 5.1.0

### Changed

- Use fft_params struct for accuracy and benchmark clients.

## hipFFT 1.0.6 for ROCm 5.0.0

### Fixed

- Fixed incorrect reporting of rocFFT version.

### Changed

- Unconditionally enabled callback functionality.  On the CUDA backend, callbacks only run
  correctly when hipFFT is built as a static library, and is linked against the static cuFFT
  library.

## hipFFT 1.0.5 for ROCm 4.5.0

### Added

- Added support for Windows 10 as a build target.

### Changed

- Packaging split into a runtime package called hipfft and a development package called hipfft-devel.
  The development package depends on runtime. The runtime package suggests the development package
  for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging
  is introduced as a deprecated feature and will be removed in a future rocm release.

## hipFFT 1.0.4 for ROCm 4.4.0

### Fixed

- Add calls to rocFFT setup/cleanup.
- Cmake fixes for clients and backend support.

### Added

- Added support for Windows 10 as a build target.

## hipFFT 1.0.3 for ROCm 4.3.0

### Fixed

- Cmake updates.

### Added

- Added callback API in hipfftXt.h header.

## hipFFT 1.0.2 for ROCm 4.2.0

- No changes.

## hipFFT 1.0.1 for ROCm 4.1.0

### Fixed

- Fix batch support for `hipfftMakePlanMany`.
- Fix work area handling during plan creation and `hipfftSetWorkArea`.
- Honour `autoAllocate` flag.

### Changed

- Testing infrastructure reuses code from [rocFFT].

[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
[hipFFT]: https://github.com/ROCmSoftwarePlatform/hipFFT
[hipfft.readthedocs.io]: https://rocfft.readthedocs.io/en/latest/
