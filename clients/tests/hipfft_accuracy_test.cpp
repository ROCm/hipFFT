// Copyright (C) 2022 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <boost/scope_exit.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hipfft.h"

#include "../hipfft_params.h"
#include "../rocFFT/clients/tests/fftw_transform.h"
#include "../rocFFT/clients/tests/rocfft_accuracy_test.h"
#include "../rocFFT/clients/tests/rocfft_against_fftw.h"
#include "../rocFFT/shared/gpubuf.h"

void fft_vs_reference(hipfft_params& params)
{
    switch(params.precision)
    {
    case fft_precision_single:
        fft_vs_reference_impl<float, hipfft_params>(params);
        break;
    case fft_precision_double:
        fft_vs_reference_impl<double, hipfft_params>(params);
        break;
    }
}

// Test for comparison between FFTW and hipFFT.
TEST_P(accuracy_test, vs_fftw)
{
    hipfft_params params(GetParam());

    params.validate();

    if(!params.valid(verbose))
    {
        if(verbose)
        {
            std::cout << "Invalid parameters, skip this test." << std::endl;
        }
        GTEST_SKIP();
    }

    fft_vs_reference(params);
    SUCCEED();
}

// Stubs for callback tests.
// Many seem to be called unconditionally, so we can't throw exceptions in
// most cases.
void* get_load_callback_host(fft_array_type itype, fft_precision precision)
{
    return nullptr;
}
void  apply_load_callback(const fft_params& params, fftw_data_t& input) {}
void  apply_store_callback(const fft_params& params, fftw_data_t& output) {}
void* get_store_callback_host(fft_array_type otype, fft_precision precision)
{
    throw std::runtime_error("get_store_callback_host not implemented");
    return nullptr;
}
