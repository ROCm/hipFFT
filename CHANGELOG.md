# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## [(Unreleased) hipFFT 1.0.4 for ROCm 4.4.0]

### Fixed
- Add calls to rocFFT setup/cleanup.

### Added

- Added support for Windows 10 as a build target.

## [hipFFT 1.0.3 for ROCm 4.3.0]

### Fixed
- Cmake updates.

### Added
- Added callback API in hipfftXt.h header.

## [hipFFT 1.0.2 for ROCm 4.2.0]

- No changes.

## [hipFFT 1.0.1 for ROCm 4.1.0]

### Fixed
- Fix batch support for `hipfftMakePlanMany`.
- Fix work area handling during plan creation and `hipfftSetWorkArea`.
- Honour `autoAllocate` flag.

### Changed
- Testing infrastructure reuses code from [rocFFT].

[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
[hipFFT]: https://github.com/ROCmSoftwarePlatform/hipFFT
[hipfft.readthedocs.io]: https://rocfft.readthedocs.io/en/latest/
