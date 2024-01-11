// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hipfft/hipfft.h"

#include "../hipfft_params.h"

#include "../../shared/accuracy_test.h"
#include "../../shared/fftw_transform.h"
#include "../../shared/gpubuf.h"
#include "../../shared/rocfft_against_fftw.h"
#include "../../shared/rocfft_complex.h"

void fft_vs_reference(hipfft_params& params, bool round_trip)
{
    switch(params.precision)
    {
    case fft_precision_half:
        fft_vs_reference_impl<_Float16, hipfft_params>(params, round_trip);
        break;
    case fft_precision_single:
        fft_vs_reference_impl<float, hipfft_params>(params, round_trip);
        break;
    case fft_precision_double:
        fft_vs_reference_impl<double, hipfft_params>(params, round_trip);
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

    // only do round trip for non-field FFTs
    bool round_trip = params.ifields.empty() && params.ofields.empty();

    if(!params.run_callbacks)
        fft_vs_reference(params, round_trip);

    SUCCEED();
}

#ifdef __HIP__

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata load_callback(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    if(input == testdata->base)
        return input[offset] * testdata->scalar;
    // wrong base address passed, return something obviously wrong
    else
    {
        // wrong base address passed, return something obviously wrong
        return input[0];
    }
}

__device__ auto load_callback_dev_half           = load_callback<_Float16>;
__device__ auto load_callback_dev_complex_half   = load_callback<rocfft_complex<_Float16>>;
__device__ auto load_callback_dev_float          = load_callback<float>;
__device__ auto load_callback_dev_complex_float  = load_callback<rocfft_complex<float>>;
__device__ auto load_callback_dev_double         = load_callback<double>;
__device__ auto load_callback_dev_complex_double = load_callback<rocfft_complex<double>>;

void* get_load_callback_host(fft_array_type itype,
                             fft_precision  precision,
                             bool           round_trip_inverse = false)
{
    void* load_callback_host = nullptr;
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            EXPECT_EQ(hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_dev_complex_half),
                                          sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_dev_complex_float),
                                          sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_dev_complex_double),
                                          sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_half), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_float), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &load_callback_host, HIP_SYMBOL(load_callback_dev_double), sizeof(void*)),
                      hipSuccess);
            return load_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return load_callback_host;
    }
}

template <typename Tdata>
__host__ __device__ static void
    store_callback(Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    if(output == testdata->base)
    {
        output[offset] = element + testdata->scalar;
    }
    // otherwise, wrong base address passed, just don't write
}

__device__ auto store_callback_dev_half           = store_callback<_Float16>;
__device__ auto store_callback_dev_complex_half   = store_callback<rocfft_complex<_Float16>>;
__device__ auto store_callback_dev_float          = store_callback<float>;
__device__ auto store_callback_dev_complex_float  = store_callback<rocfft_complex<float>>;
__device__ auto store_callback_dev_double         = store_callback<double>;
__device__ auto store_callback_dev_complex_double = store_callback<rocfft_complex<double>>;

void* get_store_callback_host(fft_array_type otype,
                              fft_precision  precision,
                              bool           round_trip_inverse = false)
{
    void* store_callback_host = nullptr;
    switch(otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_complex_half),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_complex_float),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_complex_double),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            EXPECT_EQ(hipMemcpyFromSymbol(
                          &store_callback_host, HIP_SYMBOL(store_callback_dev_half), sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_single:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_float),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        case fft_precision_double:
            EXPECT_EQ(hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_dev_double),
                                          sizeof(void*)),
                      hipSuccess);
            return store_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return store_callback_host;
    }
}

// implement result scaling as a store callback, as rocFFT tests do
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output)
{
    if(!params.run_callbacks && params.scale_factor == 1.0)
        return;

    callback_test_data cbdata;
    cbdata.scalar = params.store_cb_scalar;
    cbdata.base   = output.front().data();

    switch(params.otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(std::complex<_Float16>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<_Float16>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(std::complex<float>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<float>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(std::complex<double>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<double>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        // planar wouldn't run callbacks, but we could still want scaling
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(std::complex<_Float16>);
            for(auto& buf : output)
            {
                const size_t num_elems = buf.size() / elem_size;

                auto output_begin = reinterpret_cast<rocfft_complex<_Float16>*>(buf.data());
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(params.scale_factor != 1.0)
                        element = element * params.scale_factor;
                }
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(std::complex<float>);
            for(auto& buf : output)
            {
                const size_t num_elems = buf.size() / elem_size;

                auto output_begin = reinterpret_cast<rocfft_complex<float>*>(buf.data());
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(params.scale_factor != 1.0)
                        element = element * params.scale_factor;
                }
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(std::complex<double>);
            for(auto& buf : output)
            {
                const size_t num_elems = buf.size() / elem_size;

                auto output_begin = reinterpret_cast<rocfft_complex<double>*>(buf.data());
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(params.scale_factor != 1.0)
                        element = element * params.scale_factor;
                }
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(_Float16);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<_Float16*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                if(params.scale_factor != 1.0)
                    element = element * params.scale_factor;
                if(params.run_callbacks)
                    store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}

// apply load callback if necessary
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input)
{
    if(!params.run_callbacks)
        return;
    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.load_cb_scalar;
    cbdata.base   = input.front().data();

    switch(params.itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(std::complex<_Float16>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<_Float16>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(std::complex<float>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<float>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(std::complex<double>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<double>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(_Float16);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<_Float16*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    default:
        // this is FFTW data which should always be interleaved (if complex)
        abort();
    }
}
#else
// Stubs for callback tests.
// Many seem to be called unconditionally, so we can't throw exceptions in
// most cases.

void* get_load_callback_host(fft_array_type itype,
                             fft_precision  precision,
                             bool           round_trip_inverse = false)
{
    return nullptr;
}
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input) {}

// implement result scaling as a store callback, as rocFFT tests do
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output)
{
    if(params.scale_factor == 1.0)
        return;
    switch(params.precision)
    {
    case fft_precision_half:
    {
        const size_t elem_size = sizeof(_Float16);
        for(auto& buf : output)
        {
            const size_t num_elems    = buf.size() / elem_size;
            auto         output_begin = reinterpret_cast<_Float16*>(buf.data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                element       = static_cast<double>(element) * params.scale_factor;
            }
        }
        break;
    }
    case fft_precision_single:
    {
        const size_t elem_size = sizeof(float);
        for(auto& buf : output)
        {
            const size_t num_elems    = buf.size() / elem_size;
            auto         output_begin = reinterpret_cast<float*>(buf.data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                element       = element * params.scale_factor;
            }
        }
        break;
    }
    case fft_precision_double:
    {
        const size_t elem_size = sizeof(double);
        for(auto& buf : output)
        {
            const size_t num_elems    = buf.size() / elem_size;
            auto         output_begin = reinterpret_cast<double*>(buf.data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                element       = element * params.scale_factor;
            }
        }
        break;
    }
    }
}
void* get_store_callback_host(fft_array_type otype,
                              fft_precision  precision,
                              bool           round_trip_inverse = false)
{
    throw std::runtime_error("get_store_callback_host not implemented");
    return nullptr;
}
#endif
