# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## (Unreleased) hipFFT 1.0.8

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
