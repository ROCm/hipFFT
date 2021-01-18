# Change log for hipFFT

Partial documentation for hipFFT is available at [hipFFT].

## Unreleased

### Changed

- When building the tests or rider clients, the [rocFFT] submodule
  must be up-to-date.  To update it, run:

    git submodule update --init

### Fixes

- Fix batch support for `hipfftMakePlanMany`.

- Fix work area handling during plan creation and `hipfftSetWorkArea`.


[hipFFT]: https://github.com/ROCmSoftwarePlatform/hipFFT
[hipfft.readthedocs.io]: https://rocfft.readthedocs.io/en/latest/
[rocFFT]: https://github.com/ROCmSoftwarePlatform/rocFFT
