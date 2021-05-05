
// Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rocFFT/clients/tests/accuracy_test.h"
#include "rocFFT/clients/tests/rocfft_against_fftw.h"

#include "accuracy_test.h"

static std::vector<size_t> pow2_range  = {4, 8, 16, 32, 128};
static std::vector<size_t> pow3_range  = {3, 9, 27, 81};
static std::vector<size_t> pow5_range  = {5, 25, 125};
static std::vector<size_t> pow7_range  = {7, 49, 84};
static std::vector<size_t> mix_range   = {6, 12, 20, 30, 120, 150};
static std::vector<size_t> prime_range = {19, 37, 59, 73, 97};

static std::vector<std::vector<size_t>> vpow2_range = {pow2_range, pow2_range, pow2_range};
INSTANTIATE_TEST_SUITE_P(
    pow2_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    pow2_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow2_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow2_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow2_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow2_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow2_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));

static std::vector<std::vector<size_t>> vpow3_range = {pow3_range, pow3_range, pow3_range};
INSTANTIATE_TEST_SUITE_P(
    pow3_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    pow3_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow3_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow3_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow3_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow3_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow3_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));

static std::vector<std::vector<size_t>> vpow5_range = {pow5_range, pow5_range, pow5_range};
INSTANTIATE_TEST_SUITE_P(
    pow5_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    pow5_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow5_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow5_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow5_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow5_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow5_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));

static std::vector<std::vector<size_t>> vpow7_range = {pow7_range, pow7_range, pow7_range};
INSTANTIATE_TEST_SUITE_P(
    pow7_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow7_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    pow7_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow7_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow7_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow7_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(pow7_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vpow7_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));

static std::vector<std::vector<size_t>> vmix_range = {mix_range, mix_range, mix_range};
INSTANTIATE_TEST_SUITE_P(
    mix_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vmix_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    mix_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vmix_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(mix_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vmix_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(mix_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vmix_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));

static std::vector<std::vector<size_t>> vprime_range = {prime_range, prime_range, prime_range};
INSTANTIATE_TEST_SUITE_P(
    prime_3D_complex_forward,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vprime_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_forward),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(
    prime_3D_complex_inverse,
    hipfft_accuracy_test,
    ::testing::Combine(::testing::ValuesIn(generate_lengths(vprime_range)),
                       ::testing::ValuesIn(batch_range),
                       ::testing::ValuesIn(precision_range),
                       ::testing::Values(rocfft_transform_type_complex_inverse),
                       ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(prime_3D_real_forward,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vprime_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_forward),
                                            ::testing::ValuesIn(place_range)));

INSTANTIATE_TEST_SUITE_P(prime_3D_real_inverse,
                         hipfft_accuracy_test,
                         ::testing::Combine(::testing::ValuesIn(generate_lengths(vprime_range)),
                                            ::testing::ValuesIn(batch_range),
                                            ::testing::ValuesIn(precision_range),
                                            ::testing::Values(rocfft_transform_type_real_inverse),
                                            ::testing::ValuesIn(place_range)));
