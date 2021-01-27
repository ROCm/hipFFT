# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## [(Unreleased) hipFFT 1.0.2 for ROCm 4.2.0]

## [hipFFT 1.0.1 for ROCm 4.1.0]

### Fixes
- Fix batch support for `hipfftMakePlanMany`.
- Fix work area handling during plan creation and `hipfftSetWorkArea`.
- Honour `autoAllocate` flag.

### Changes
- Testing infrastructure reuses code from [rocFFT].

[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
[hipFFT]: https://github.com/ROCmSoftwarePlatform/hipFFT
[hipfft.readthedocs.io]: https://rocfft.readthedocs.io/en/latest/
