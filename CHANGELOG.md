# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## hipFFT 1.0.5 for ROCm 4.5.0

### Fixed
- Add calls to rocFFT setup/cleanup.
- Cmake fixes for clients and backend support.

### Added
- Added support for Windows 10 as a build target.

### Changed
- Packaging split into a runtime package called hipfft and a development package called hipfft-devel.
  The development package depends on runtime. The runtime package suggests the development package
  for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging
  is introduced as a deprecated feature and will be removed in a future rocm release.

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
