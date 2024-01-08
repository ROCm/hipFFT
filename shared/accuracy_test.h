// Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#ifndef ACCURACY_TEST
#define ACCURACY_TEST

#include <algorithm>
#include <functional>
#include <future>
#include <iterator>
#include <string>
#include <vector>

#include "enum_to_string.h"
#include "fft_params.h"
#include "fftw_transform.h"
#include "gpubuf.h"
#include "rocfft_against_fftw.h"
#include "test_params.h"

extern int    verbose;
extern size_t ramgb;
extern bool   fftw_compare;

static const size_t ONE_GiB = 1 << 30;

inline size_t bytes_to_GiB(const size_t bytes)
{
    return bytes == 0 ? 0 : (bytes - 1 + ONE_GiB) / ONE_GiB;
}

typedef std::tuple<fft_transform_type, fft_result_placement, fft_array_type, fft_array_type>
    type_place_io_t;

// Remember the results of the last FFT we computed with FFTW.  Tests
// are ordered so that later cases can often reuse this result.
struct last_cpu_fft_cache
{
    // keys to the cache
    std::vector<size_t> length;
    size_t              nbatch         = 0;
    fft_transform_type  transform_type = fft_transform_type_complex_forward;
    bool                run_callbacks  = false;
    fft_precision       precision      = fft_precision_single;

    // FFTW input/output
    std::vector<hostbuf> cpu_input;
    std::vector<hostbuf> cpu_output;
};
extern last_cpu_fft_cache last_cpu_fft_data;

struct system_memory
{
    size_t total_bytes = 0;
    size_t free_bytes  = 0;
};
extern system_memory start_memory;

system_memory get_system_memory();

// Estimate the amount of host memory needed for buffers.
inline size_t needed_ram_buffers(const fft_params& params, const int verbose)
{
    // This calculation is assuming contiguous data but noncontiguous buffers
    // are assumed to require a close enough amount of space for the purposes
    // of this estimate.

    size_t needed_ram = 6
                        * std::accumulate(params.length.begin(),
                                          params.length.end(),
                                          static_cast<size_t>(1),
                                          std::multiplies<size_t>());

    // Account for precision and data type:
    if(params.transform_type != fft_transform_type_real_forward
       && params.transform_type != fft_transform_type_real_inverse)
    {
        needed_ram *= 2;
    }
    switch(params.precision)
    {
    case fft_precision_half:
        needed_ram *= 2;
        break;
    case fft_precision_single:
        needed_ram *= 4;
        break;
    case fft_precision_double:
        needed_ram *= 8;
        break;
    }

    needed_ram *= params.nbatch;

    if(verbose)
    {
        std::cout << "required host memory for buffers (GiB): " << bytes_to_GiB(needed_ram) << "\n";
    }

    return needed_ram;
}

template <typename Tfloat>
bool fftw_plan_uses_bluestein(const typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan)
{
#ifdef FFTW_HAVE_SPRINT_PLAN
    char*       print_plan_c_str = fftw_sprint_plan<Tfloat>(cpu_plan);
    std::string print_plan(print_plan_c_str);
    free(print_plan_c_str);
    return print_plan.find("bluestein") != std::string::npos;
#else
    // assume worst case (bluestein is always used)
    return true;
#endif
}

// Estimate the amount of host memory needed for fftw.
template <typename Tfloat>
inline size_t needed_ram_fftw(const fft_params&                                  contiguous_params,
                              const typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan,
                              const int                                          verbose)
{
    size_t total_length = std::accumulate(contiguous_params.length.begin(),
                                          contiguous_params.length.end(),
                                          static_cast<size_t>(1),
                                          std::multiplies<size_t>());
    size_t needed_ram   = 0;
    // Detect Bluestein in plan
    if(fftw_plan_uses_bluestein<Tfloat>(cpu_plan))
    {
        for(size_t dim : contiguous_params.length)
        {
            unsigned int needed_ram_dim = dim;

            // Next-plus-one-power-of-two multiplied any other lengths
            needed_ram_dim--;

            needed_ram_dim |= needed_ram_dim >> 2;
            needed_ram_dim |= needed_ram_dim >> 4;
            needed_ram_dim |= needed_ram_dim >> 8;
            needed_ram_dim |= needed_ram_dim >> 16;

            needed_ram_dim++;

            needed_ram_dim *= 2 * (total_length / dim);

            if(needed_ram_dim > needed_ram)
            {
                needed_ram = needed_ram_dim;
            }
        }
    }

    // Account for precision and data type:
    if(contiguous_params.transform_type != fft_transform_type_real_forward
       && contiguous_params.transform_type != fft_transform_type_real_inverse)
    {
        needed_ram *= 2;
    }
    switch(contiguous_params.precision)
    {
    case fft_precision_half:
        needed_ram *= 2;
        break;
    case fft_precision_single:
        needed_ram *= 4;
        break;
    case fft_precision_double:
        needed_ram *= 8;
        break;
    }

    needed_ram *= contiguous_params.nbatch;

    if(verbose)
    {
        std::cout << "required host memory for FFTW (GiB): " << bytes_to_GiB(needed_ram) << "\n";
    }

    return needed_ram;
}

// Base gtest class for comparison with FFTW.
class accuracy_test : public ::testing::TestWithParam<fft_params>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<accuracy_test::ParamType>& info)
    {
        return info.param.token();
    }
};

const static std::vector<size_t> batch_range = {2, 1};

const static std::vector<fft_precision> precision_range_full
    = {fft_precision_double, fft_precision_single, fft_precision_half};
const static std::vector<fft_precision> precision_range_sp_dp
    = {fft_precision_double, fft_precision_single};

const static std::vector<fft_result_placement> place_range
    = {fft_placement_inplace, fft_placement_notinplace};
const static std::vector<fft_transform_type> trans_type_range
    = {fft_transform_type_complex_forward, fft_transform_type_real_forward};
const static std::vector<fft_transform_type> trans_type_range_complex
    = {fft_transform_type_complex_forward};
const static std::vector<fft_transform_type> trans_type_range_real
    = {fft_transform_type_real_forward};

// Given a vector of vector of lengths, generate all unique permutations.
// Add an optional vector of ad-hoc lengths to the result.
inline std::vector<std::vector<size_t>>
    generate_lengths(const std::vector<std::vector<size_t>>& inlengths)
{
    std::vector<std::vector<size_t>> output;
    if(inlengths.size() == 0)
    {
        return output;
    }
    const size_t        dim = inlengths.size();
    std::vector<size_t> looplength(dim);
    for(unsigned int i = 0; i < dim; ++i)
    {
        looplength[i] = inlengths[i].size();
    }
    for(unsigned int idx = 0; idx < inlengths.size(); ++idx)
    {
        std::vector<size_t> index(dim);
        do
        {
            std::vector<size_t> length(dim);
            for(unsigned int i = 0; i < dim; ++i)
            {
                length[i] = inlengths[i][index[i]];
            }
            output.push_back(length);
        } while(increment_rowmajor(index, looplength));
    }
    // uniquify the result
    std::sort(output.begin(), output.end());
    output.erase(std::unique(output.begin(), output.end()), output.end());
    return output;
}

// Return the valid rocFFT input and output types for a given transform type.
inline std::vector<std::pair<fft_array_type, fft_array_type>>
    iotypes(const fft_transform_type   transformType,
            const fft_result_placement place,
            const bool                 planar = true)
{
    std::vector<std::pair<fft_array_type, fft_array_type>> iotypes;
    switch(transformType)
    {
    case fft_transform_type_complex_forward:
    case fft_transform_type_complex_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_complex_interleaved, fft_array_type_complex_interleaved));
        if(planar)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_complex_planar, fft_array_type_complex_planar));
            if(place == fft_placement_notinplace)
            {
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_planar, fft_array_type_complex_interleaved));
                iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                    fft_array_type_complex_interleaved, fft_array_type_complex_planar));
            }
        }
        break;
    case fft_transform_type_real_forward:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_real, fft_array_type_hermitian_interleaved));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_real, fft_array_type_hermitian_planar));
        }
        break;
    case fft_transform_type_real_inverse:
        iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
            fft_array_type_hermitian_interleaved, fft_array_type_real));
        if(planar && place == fft_placement_notinplace)
        {
            iotypes.push_back(std::make_pair<fft_array_type, fft_array_type>(
                fft_array_type_hermitian_planar, fft_array_type_real));
        }
        break;
    default:
        throw std::runtime_error("Invalid transform type");
    }
    return iotypes;
}

// Generate all combinations of input/output types, from combinations of transform and placement
// types.
static std::vector<type_place_io_t>
    generate_types(fft_transform_type                       transform_type,
                   const std::vector<fft_result_placement>& place_range,
                   const bool                               planar)
{
    std::vector<type_place_io_t> ret;
    for(auto place : place_range)
    {
        for(auto iotype : iotypes(transform_type, place, planar))
        {
            ret.push_back(std::make_tuple(transform_type, place, iotype.first, iotype.second));
        }
    }
    return ret;
}

struct stride_generator
{
    struct stride_dist
    {
        stride_dist(const std::vector<size_t>& s, size_t d)
            : stride(s)
            , dist(d)
        {
        }
        std::vector<size_t> stride;
        size_t              dist;
    };

    // NOTE: allow for this ctor to be implicit, so it's less typing for a test writer
    //
    // cppcheck-suppress noExplicitConstructor
    stride_generator(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_list(stride_list_in)
    {
    }
    virtual std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                              size_t                     batch) const
    {
        std::vector<stride_dist> ret;
        for(const auto& s : stride_list)
            ret.emplace_back(s, 0);
        return ret;
    }
    std::vector<std::vector<size_t>> stride_list;
};

// Generate strides such that batch is essentially the innermost dimension
// e.g. given a batch-2 4x3x2 transform which logically looks like:
//
// batch0:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// batch1:
// A B A B
// A B A B
// A B A B
//
// A B A B
// A B A B
// A B A B
//
// we instead do stride-2 4x3x2 transform where first batch is the
// A's and second batch is the B's.
struct stride_generator_3D_inner_batch : public stride_generator
{
    explicit stride_generator_3D_inner_batch(const std::vector<std::vector<size_t>>& stride_list_in)
        : stride_generator(stride_list_in)
    {
    }
    std::vector<stride_dist> generate(const std::vector<size_t>& lengths,
                                      size_t                     batch) const override
    {
        std::vector<stride_dist> ret = stride_generator::generate(lengths, batch);
        std::vector<size_t> strides{lengths[1] * lengths[2] * batch, lengths[2] * batch, batch};
        ret.emplace_back(strides, 1);
        return ret;
    }
};

// Create an array of parameters to pass to gtest.  Base generator
// that allows choosing transform type.
inline auto param_generator_base(const std::vector<fft_transform_type>&   type_range,
                                 const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 decltype(generate_types)                 types_generator,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,
                                 const bool                               planar        = true,
                                 const bool                               run_callbacks = false)
{

    std::vector<fft_params> params;

    // For any length, we compute double-precision CPU reference
    // for largest batch size first and reuse for smaller batch
    // sizes, then convert to single-precision.

    for(auto& transform_type : type_range)
    {
        for(const auto& lengths : v_lengths)
        {
            // try to ensure that we are given literal lengths, not
            // something to be passed to generate_lengths
            if(lengths.empty() || lengths.size() > 3)
            {
                continue;
            }
            {
                for(const auto precision : precision_range)
                {
                    for(const auto batch : batch_range)
                    {
                        for(const auto& types :
                            types_generator(transform_type, place_range, planar))
                        {
                            for(const auto& istride_dist : istride.generate(lengths, batch))
                            {
                                for(const auto& ostride_dist : ostride.generate(lengths, batch))
                                {
                                    for(const auto& ioffset : ioffset_range)
                                    {
                                        for(const auto& ooffset : ooffset_range)
                                        {
                                            fft_params param;

                                            param.length         = lengths;
                                            param.istride        = istride_dist.stride;
                                            param.ostride        = ostride_dist.stride;
                                            param.nbatch         = batch;
                                            param.precision      = precision;
                                            param.transform_type = std::get<0>(types);
                                            param.placement      = std::get<1>(types);
                                            param.idist          = istride_dist.dist;
                                            param.odist          = ostride_dist.dist;
                                            param.itype          = std::get<2>(types);
                                            param.otype          = std::get<3>(types);
                                            param.ioffset        = ioffset;
                                            param.ooffset        = ooffset;

                                            if(run_callbacks)
                                            {
                                                // add a test if both input and output support callbacks
                                                if(param.itype != fft_array_type_complex_planar
                                                   && param.itype != fft_array_type_hermitian_planar
                                                   && param.otype != fft_array_type_complex_planar
                                                   && param.otype
                                                          != fft_array_type_hermitian_planar)
                                                {
                                                    param.run_callbacks = true;
                                                }
                                                else
                                                {
                                                    continue;
                                                }
                                            }
                                            param.validate();

                                            // Keeping the random number generator here
                                            // allows one to run the same tests for a given
                                            // random seed; ie the test suite is repeatable.
                                            std::hash<std::string>           hasher;
                                            std::ranlux24_base               gen(random_seed
                                                                   + hasher(param.token()));
                                            std::uniform_real_distribution<> dis(0.0, 1.0);

                                            if(param.is_planar())
                                            {
                                                const double roll = dis(gen);
                                                if(roll > planar_prob)
                                                {
                                                    if(verbose > 4)
                                                    {
                                                        std::cout << "Planar transform skipped "
                                                                     "(planar_prob: "
                                                                  << planar_prob << " > " << roll
                                                                  << ")\n";
                                                    }
                                                    continue;
                                                }
                                            }
                                            if(run_callbacks)
                                            {
                                                const double roll = dis(gen);
                                                if(roll > callback_prob)
                                                {

                                                    if(verbose > 4)
                                                    {
                                                        std::cout << "Callback transform skipped "
                                                                     "(planar_prob: "
                                                                  << planar_prob << " > " << roll
                                                                  << ")\n";
                                                    }
                                                    continue;
                                                }
                                            }

                                            if(param.valid(0))
                                            {
                                                params.push_back(param);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return params;
}

// Create an array of parameters to pass to gtest.  Default generator
// that picks all transform types.
inline auto param_generator(const std::vector<std::vector<size_t>>&  v_lengths,
                            const std::vector<fft_precision>&        precision_range,
                            const std::vector<size_t>&               batch_range,
                            const stride_generator&                  istride,
                            const stride_generator&                  ostride,
                            const std::vector<std::vector<size_t>>&  ioffset_range,
                            const std::vector<std::vector<size_t>>&  ooffset_range,
                            const std::vector<fft_result_placement>& place_range,
                            const bool                               planar,
                            const bool                               run_callbacks = false)
{
    return param_generator_base(trans_type_range,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,
                                planar,
                                run_callbacks);
}

// Create an array of parameters to pass to gtest.  Only tests complex-type transforms
inline auto param_generator_complex(const std::vector<std::vector<size_t>>&  v_lengths,
                                    const std::vector<fft_precision>&        precision_range,
                                    const std::vector<size_t>&               batch_range,
                                    const stride_generator&                  istride,
                                    const stride_generator&                  ostride,
                                    const std::vector<std::vector<size_t>>&  ioffset_range,
                                    const std::vector<std::vector<size_t>>&  ooffset_range,
                                    const std::vector<fft_result_placement>& place_range,
                                    const bool                               planar,
                                    const bool                               run_callbacks = false)
{
    return param_generator_base(trans_type_range_complex,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,
                                planar,
                                run_callbacks);
}

// Create an array of parameters to pass to gtest.
inline auto param_generator_real(const std::vector<std::vector<size_t>>&  v_lengths,
                                 const std::vector<fft_precision>&        precision_range,
                                 const std::vector<size_t>&               batch_range,
                                 const stride_generator&                  istride,
                                 const stride_generator&                  ostride,
                                 const std::vector<std::vector<size_t>>&  ioffset_range,
                                 const std::vector<std::vector<size_t>>&  ooffset_range,
                                 const std::vector<fft_result_placement>& place_range,
                                 const bool                               planar,
                                 const bool                               run_callbacks = false)
{
    return param_generator_base(trans_type_range_real,
                                v_lengths,
                                precision_range,
                                batch_range,
                                generate_types,
                                istride,
                                ostride,
                                ioffset_range,
                                ooffset_range,
                                place_range,
                                planar,
                                run_callbacks);
}

template <class Tcontainer>
auto param_generator_token(const Tcontainer& tokens)
{
    std::vector<fft_params> params;
    params.reserve(tokens.size());
    for(auto t : tokens)
    {
        params.push_back({});
        params.back().from_token(t);
    }
    return params;
}

struct callback_test_data
{
    // scalar to modify the input/output with
    double scalar;
    // base address of input, to ensure that each callback gets an offset from that base
    void* base;
};

void* get_load_callback_host(fft_array_type itype,
                             fft_precision  precision,
                             bool           round_trip_inverse);
void  apply_load_callback(const fft_params& params, std::vector<hostbuf>& input);
void  apply_store_callback(const fft_params& params, std::vector<hostbuf>& output);
void* get_store_callback_host(fft_array_type otype,
                              fft_precision  precision,
                              bool           round_trip_inverse);

static auto allocate_cpu_fft_buffer(const fft_precision        precision,
                                    const fft_array_type       type,
                                    const std::vector<size_t>& size)
{
    // FFTW does not support half-precision, so we do single instead.
    // So if we need to do a half-precision FFTW transform, allocate
    // enough buffer for single-precision instead.
    return allocate_host_buffer(
        precision == fft_precision_half ? fft_precision_single : precision, type, size);
}

template <typename Tfloat>
inline void execute_cpu_fft(fft_params&                                  params,
                            fft_params&                                  contiguous_params,
                            typename fftw_trait<Tfloat>::fftw_plan_type& cpu_plan,
                            std::vector<hostbuf>&                        cpu_input,
                            std::vector<hostbuf>&                        cpu_output)
{
    // CPU output might not be allocated already for us, if FFTW never
    // needed an output buffer during planning
    if(cpu_output.empty())
        cpu_output = allocate_cpu_fft_buffer(
            contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);

    // If this is either C2R or callbacks are enabled, the
    // input will be modified.  So we need to modify the copy instead.
    std::vector<hostbuf>  cpu_input_copy(cpu_input.size());
    std::vector<hostbuf>* input_ptr = &cpu_input;
    if(params.run_callbacks || contiguous_params.transform_type == fft_transform_type_real_inverse)
    {
        for(size_t i = 0; i < cpu_input.size(); ++i)
        {
            cpu_input_copy[i] = cpu_input[i].copy();
        }

        input_ptr = &cpu_input_copy;
    }

    // run FFTW (which may destroy CPU input)
    apply_load_callback(params, *input_ptr);
    fftw_run<Tfloat>(contiguous_params.transform_type, cpu_plan, *input_ptr, cpu_output);
    // clean up
    fftw_destroy_plan_type(cpu_plan);
    // ask FFTW to fully clean up, since it tries to cache plan details
    fftw_cleanup();
    cpu_plan = nullptr;
    apply_store_callback(params, cpu_output);
}

// execute the GPU transform
template <class Tparams>
inline void execute_gpu_fft(Tparams&              params,
                            std::vector<void*>&   pibuffer,
                            std::vector<void*>&   pobuffer,
                            std::vector<gpubuf>&  obuffer,
                            std::vector<hostbuf>& gpu_output,
                            bool                  round_trip_inverse = false)
{
    gpubuf_t<callback_test_data> load_cb_data_dev;
    gpubuf_t<callback_test_data> store_cb_data_dev;
    if(params.run_callbacks)
    {
        void* load_cb_host
            = get_load_callback_host(params.itype, params.precision, round_trip_inverse);

        callback_test_data load_cb_data_host;

        if(round_trip_inverse)
        {
            load_cb_data_host.scalar = params.store_cb_scalar;
        }
        else
        {
            load_cb_data_host.scalar = params.load_cb_scalar;
        }

        load_cb_data_host.base = pibuffer.front();

        auto hip_status = hipSuccess;

        hip_status = load_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP();
            }
            else
            {
                GTEST_FAIL();
            }
        }
        hip_status = hipMemcpy(load_cb_data_dev.data(),
                               &load_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP();
            }
            else
            {
                GTEST_FAIL();
            }
        }

        void* store_cb_host
            = get_store_callback_host(params.otype, params.precision, round_trip_inverse);

        callback_test_data store_cb_data_host;

        if(round_trip_inverse)
        {
            store_cb_data_host.scalar = params.load_cb_scalar;
        }
        else
        {
            store_cb_data_host.scalar = params.store_cb_scalar;
        }

        store_cb_data_host.base = pobuffer.front();

        hip_status = store_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP();
            }
            else
            {
                GTEST_FAIL();
            }
        }

        hip_status = hipMemcpy(store_cb_data_dev.data(),
                               &store_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP();
            }
            else
            {
                GTEST_FAIL();
            }
        }

        auto fft_status = params.set_callbacks(
            load_cb_host, load_cb_data_dev.data(), store_cb_host, store_cb_data_dev.data());
        if(fft_status != fft_status_success)
            throw std::runtime_error("set callback failure");
    }

    // Execute the transform:
    auto fft_status = params.execute(pibuffer.data(), pobuffer.data());
    if(fft_status != fft_status_success)
        throw std::runtime_error("rocFFT plan execution failure");

    // if not comparing, then just executing the GPU FFT is all we
    // need to do
    if(!fftw_compare)
        return;

    // finalize a multi-GPU transform
    params.multi_gpu_finalize(obuffer, pobuffer);

    ASSERT_TRUE(!gpu_output.empty()) << "no output buffers";
    for(unsigned int idx = 0; idx < gpu_output.size(); ++idx)
    {
        ASSERT_TRUE(gpu_output[idx].data() != nullptr)
            << "output buffer index " << idx << " is empty";
        auto hip_status = hipMemcpy(gpu_output[idx].data(),
                                    pobuffer.at(idx),
                                    gpu_output[idx].size(),
                                    hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
        {
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP() << "hipMemcpy failure";
            }
            else
            {
                GTEST_FAIL() << "hipMemcpy failure";
            }
        }
    }
    if(verbose > 2)
    {
        std::cout << "GPU output:\n";
        params.print_obuffer(gpu_output);
    }
    if(verbose > 5)
    {
        std::cout << "flat GPU output:\n";
        params.print_obuffer_flat(gpu_output);
    }
}

template <typename Tfloat>
static void assert_init_value(const std::vector<hostbuf>& output,
                              const size_t                idx,
                              const Tfloat                orig_value);

template <>
void assert_init_value(const std::vector<hostbuf>& output, const size_t idx, const float orig_value)
{
    float actual_value = reinterpret_cast<const float*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const double                orig_value)
{
    double actual_value = reinterpret_cast<const double*>(output.front().data())[idx];
    ASSERT_EQ(actual_value, orig_value) << "index " << idx;
}

template <>
void assert_init_value(const std::vector<hostbuf>& output,
                       const size_t                idx,
                       const rocfft_complex<float> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<float> actual_value
            = reinterpret_cast<const rocfft_complex<float>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<float> actual_value{
            reinterpret_cast<const float*>(output.front().data())[idx],
            reinterpret_cast<const float*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

template <>
void assert_init_value(const std::vector<hostbuf>&  output,
                       const size_t                 idx,
                       const rocfft_complex<double> orig_value)
{
    // if this is interleaved, check directly
    if(output.size() == 1)
    {
        rocfft_complex<double> actual_value
            = reinterpret_cast<const rocfft_complex<double>*>(output.front().data())[idx];
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
    else
    {
        // planar
        rocfft_complex<double> actual_value{
            reinterpret_cast<const double*>(output.front().data())[idx],
            reinterpret_cast<const double*>(output.back().data())[idx]};
        ASSERT_EQ(actual_value.x, orig_value.x) << "x index " << idx;
        ASSERT_EQ(actual_value.y, orig_value.y) << "y index " << idx;
    }
}

static const int OUTPUT_INIT_PATTERN = 0xcd;
template <class Tfloat>
void check_single_output_stride(const std::vector<hostbuf>& output,
                                const size_t                offset,
                                const std::vector<size_t>&  length,
                                const std::vector<size_t>&  stride,
                                const size_t                i)
{
    Tfloat orig;
    memset(static_cast<void*>(&orig), OUTPUT_INIT_PATTERN, sizeof(Tfloat));

    size_t curLength         = length[i];
    size_t curStride         = stride[i];
    size_t nextSmallerLength = i == length.size() - 1 ? 0 : length[i + 1];
    size_t nextSmallerStride = i == stride.size() - 1 ? 0 : stride[i + 1];

    if(nextSmallerLength == 0)
    {
        // this is the fastest dim, indexes that are not multiples of
        // the stride should be the initial value
        for(size_t idx = 0; idx < (curLength - 1) * curStride; ++idx)
        {
            if(idx % curStride != 0)
                assert_init_value<Tfloat>(output, idx, orig);
        }
    }
    else
    {
        for(size_t lengthIdx = 0; lengthIdx < curLength; ++lengthIdx)
        {
            // check that the space after the next smaller dim and the
            // end of this dim is initial value
            for(size_t idx = nextSmallerLength * nextSmallerStride; idx < curStride; ++idx)
                assert_init_value<Tfloat>(output, idx, orig);

            check_single_output_stride<Tfloat>(
                output, offset + lengthIdx * curStride, length, stride, i + 1);
        }
    }
}

template <class Tparams>
void check_output_strides(const std::vector<hostbuf>& output, Tparams& params)
{
    // treat batch+dist like highest length+stride, if batch > 1
    std::vector<size_t> length;
    std::vector<size_t> stride;
    if(params.nbatch > 1)
    {
        length.push_back(params.nbatch);
        stride.push_back(params.odist);
    }

    auto olength = params.olength();
    std::copy(olength.begin(), olength.end(), std::back_inserter(length));
    std::copy(params.ostride.begin(), params.ostride.end(), std::back_inserter(stride));

    if(params.precision == fft_precision_single)
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<float>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<float>>(output, 0, length, stride, 0);
    }
    else
    {
        if(params.otype == fft_array_type_real)
            check_single_output_stride<double>(output, 0, length, stride, 0);
        else
            check_single_output_stride<rocfft_complex<double>>(output, 0, length, stride, 0);
    }
}

// run rocFFT inverse transform
template <class Tparams>
inline void run_round_trip_inverse(Tparams&              params,
                                   std::vector<gpubuf>&  obuffer,
                                   std::vector<void*>&   pibuffer,
                                   std::vector<void*>&   pobuffer,
                                   std::vector<hostbuf>& gpu_output)
{
    params.validate();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    // Create FFT plan - this will also allocate work buffer, but will throw a
    // specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        std::stringstream ss;
        ss << "Failed to allocate work buffer (size: " << params.workbuffersize << ")";
        ++n_hip_failures;
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << ss.str();
        }
        else
        {
            GTEST_FAIL() << ss.str();
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "round trip inverse plan creation failed";

    auto obuffer_sizes = params.obuffer_sizes();

    if(params.placement != fft_placement_inplace)
    {
        for(unsigned int i = 0; i < obuffer_sizes.size(); ++i)
        {
            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                auto hip_status
                    = hipMemset(obuffer[i].data(), OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    if(skip_runtime_fails)
                    {
                        GTEST_SKIP() << "hipMemset failure";
                    }
                    else
                    {
                        GTEST_FAIL() << "hipMemset failure";
                    }
                }
            }
        }
    }

    // execute GPU transform
    execute_gpu_fft(params, pibuffer, pobuffer, obuffer, gpu_output, true);
}

// compare rocFFT inverse transform with forward transform input
template <class Tparams>
inline void compare_round_trip_inverse(Tparams&              params,
                                       fft_params&           contiguous_params,
                                       std::vector<hostbuf>& gpu_output,
                                       std::vector<hostbuf>& cpu_input,
                                       const VectorNorms&    cpu_input_norm,
                                       size_t                total_length)
{
    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm = std::async(std::launch::async, [&]() {
        return norm(gpu_output,
                    params.olength(),
                    params.nbatch,
                    params.precision,
                    params.otype,
                    params.ostride,
                    params.odist,
                    params.ooffset);
    });

    // compare GPU inverse output to CPU forward input
    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    const double linf_cutoff
        = type_epsilon(params.precision) * cpu_input_norm.l_inf * log(total_length);

    VectorNorms diff = distance(cpu_input,
                                gpu_output,
                                params.olength(),
                                params.nbatch,
                                params.precision,
                                contiguous_params.itype,
                                contiguous_params.istride,
                                contiguous_params.idist,
                                params.otype,
                                params.ostride,
                                params.odist,
                                linf_failures.get(),
                                linf_cutoff,
                                {0},
                                params.ooffset,
                                1.0 / total_length);

    if(verbose > 1)
    {
        std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
        std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
        std::cout << "GPU linf norm failures:";
        std::sort(linf_failures->begin(), linf_failures->end());
        for(const auto& i : *linf_failures)
        {
            std::cout << " (" << i.first << "," << i.second << ")";
        }
        std::cout << std::endl;
    }

    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
    EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half
            = std::max(max_linf_eps_half, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_half
            = std::max(max_l2_eps_half, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_single
            = std::max(max_l2_eps_single, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_input_norm.l_inf / log(total_length));
        max_l2_eps_double
            = std::max(max_l2_eps_double, diff.l_2 / cpu_input_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    EXPECT_TRUE(diff.l_inf <= linf_cutoff)
        << "Linf test failed.  Linf:" << diff.l_inf
        << "\tnormalized Linf: " << diff.l_inf / cpu_input_norm.l_inf << "\tcutoff: " << linf_cutoff
        << params.str();

    EXPECT_TRUE(diff.l_2 / cpu_input_norm.l_2
                < sqrt(log2(total_length)) * type_epsilon(params.precision))
        << "L2 test failed. L2: " << diff.l_2
        << "\tnormalized L2: " << diff.l_2 / cpu_input_norm.l_2
        << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
        << params.str();
}

// RAII type to put data into the cache when this object leaves scope
struct StoreCPUDataToCache
{
    StoreCPUDataToCache(std::vector<hostbuf>& cpu_input, std::vector<hostbuf>& cpu_output)
        : cpu_input(cpu_input)
        , cpu_output(cpu_output)
    {
    }
    ~StoreCPUDataToCache()
    {
        last_cpu_fft_data.cpu_output.swap(cpu_output);
        last_cpu_fft_data.cpu_input.swap(cpu_input);
    }
    std::vector<hostbuf>& cpu_input;
    std::vector<hostbuf>& cpu_output;
};

// run CPU + rocFFT transform with the given params and compare
template <class Tfloat, class Tparams>
inline void fft_vs_reference_impl(Tparams& params, bool round_trip)
{
    // Call hipGetLastError to reset any errors
    // returned by previous HIP runtime API calls.
    hipError_t hip_status = hipGetLastError();

    // Make sure that the parameters make sense:
    ASSERT_TRUE(params.valid(verbose));

    size_t needed_ram = needed_ram_buffers(params, verbose);

    if(ramgb > 0 && needed_ram > ramgb * ONE_GiB)
    {
        GTEST_SKIP() << "needed_ramgb: " << bytes_to_GiB(needed_ram) << ", ramgb limit: " << ramgb
                     << ".\n";
    }

    auto ibuffer_sizes = params.ibuffer_sizes();
    auto obuffer_sizes = params.obuffer_sizes();

    size_t vram_avail = 0;

    if(vramgb == 0)
    {
        // Check free and total available memory:
        size_t free       = 0;
        size_t total      = 0;
        auto   hip_status = hipMemGetInfo(&free, &total);
        if(hip_status != hipSuccess || total == 0)
        {
            ++n_hip_failures;
            std::stringstream ss;
            if(total == 0)
                ss << "hipMemGetInfo claims there there isn't any vram";
            else
                ss << "hipMemGetInfo failure with error " << hip_status;
            if(skip_runtime_fails)
            {
                GTEST_SKIP() << ss.str();
            }
            else
            {
                GTEST_FAIL() << ss.str();
            }
        }
        vram_avail = total;
    }
    else
    {
        vram_avail = vramgb * ONE_GiB;
    }

    // First try a quick estimation of vram footprint, to speed up skipping tests
    // that are too large to fit in the gpu (no plan created with the rocFFT backend)
    const auto raw_vram_footprint
        = params.fft_params_vram_footprint() + twiddle_table_vram_footprint(params);

    if(!vram_fits_problem(raw_vram_footprint, vram_avail))
    {
        GTEST_SKIP() << "Raw problem size (" << bytes_to_GiB(raw_vram_footprint)
                     << " GiB) raw data too large for device";
    }

    if(verbose > 2)
    {
        std::cout << "Raw problem size: " << raw_vram_footprint << std::endl;
    }

    // If it passed the quick estimation test, go for the more
    // accurate calculation that actually creates the plan and
    // take into account the work buffer size
    const auto vram_footprint = params.vram_footprint();
    if(!vram_fits_problem(vram_footprint, vram_avail))
    {
        if(verbose)
        {
            std::cout << "Problem raw data won't fit on device; skipped." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << bytes_to_GiB(vram_footprint)
                     << " GiB) raw data too large for device";
    }

    // Create FFT plan - this will also allocate work buffer, but
    // will throw a specific exception if that step fails
    auto plan_status = fft_status_success;
    try
    {
        plan_status = params.create_plan();
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        ++n_hip_failures;
        std::stringstream ss;
        ss << "Work buffer allocation failed with size: " << params.workbuffersize;
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << ss.str();
        }
        else
        {
            GTEST_FAIL() << ss.str();
        }
    }
    ASSERT_EQ(plan_status, fft_status_success) << "plan creation failed";

    if(!vram_fits_problem(vram_footprint, vram_avail))
    {
        if(verbose)
        {
            std::cout << "Problem won't fit on device; skipped." << std::endl;
        }
        GTEST_SKIP() << "Problem size (" << vram_footprint << ") too large for device";
        return;
    }

    fft_params contiguous_params;
    contiguous_params.length         = params.length;
    contiguous_params.precision      = params.precision;
    contiguous_params.placement      = fft_placement_notinplace;
    contiguous_params.transform_type = params.transform_type;
    contiguous_params.nbatch         = params.nbatch;
    contiguous_params.itype          = contiguous_itype(params.transform_type);
    contiguous_params.otype          = contiguous_otype(contiguous_params.transform_type);

    contiguous_params.validate();

    if(!contiguous_params.valid(verbose))
    {
        throw std::runtime_error("Invalid contiguous params");
    }

    if(verbose > 3)
    {
        std::cout << "CPU params:\n";
        std::cout << contiguous_params.str("\n\t") << std::endl;
    }

    std::vector<gpubuf> ibuffer(ibuffer_sizes.size());
    std::vector<void*>  pibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = ibuffer[i].alloc(ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            std::stringstream ss;
            ss << "hipMalloc failure for input buffer " << i << " size " << ibuffer_sizes[i] << "("
               << bytes_to_GiB(ibuffer_sizes[i]) << " GiB)"
               << " with code " << hipError_to_string(hip_status);
            ++n_hip_failures;
            if(skip_runtime_fails)
            {
                GTEST_SKIP() << ss.str();
            }
            else
            {
                GTEST_FAIL() << ss.str();
            }
        }
        pibuffer[i] = ibuffer[i].data();
    }

    // allocation counts in elements, ibuffer_sizes is in bytes
    auto ibuffer_sizes_elems = ibuffer_sizes;
    for(auto& buf : ibuffer_sizes_elems)
        buf /= var_size<size_t>(params.precision, params.itype);

    // Check cache first - nbatch is a >= comparison because we compute
    // the largest batch size and cache it.  Smaller batch runs can
    // compare against the larger data.
    std::vector<hostbuf>                 cpu_input;
    std::vector<hostbuf>                 cpu_output;
    std::shared_future<void>             convert_cpu_output_precision;
    std::shared_future<void>             convert_cpu_input_precision;
    bool                                 run_fftw = true;
    std::unique_ptr<StoreCPUDataToCache> store_to_cache;
    if(fftw_compare && last_cpu_fft_data.length == params.length
       && last_cpu_fft_data.transform_type == params.transform_type
       && last_cpu_fft_data.run_callbacks == params.run_callbacks)
    {
        if(last_cpu_fft_data.nbatch >= params.nbatch)
        {
            // use the cached input/output
            cpu_input.swap(last_cpu_fft_data.cpu_input);
            cpu_output.swap(last_cpu_fft_data.cpu_output);
            run_fftw = false;

            store_to_cache = std::make_unique<StoreCPUDataToCache>(cpu_input, cpu_output);

            if(params.precision != last_cpu_fft_data.precision)
            {
                // Tests should be ordered so we do wider first, then narrower.
                switch(params.precision)
                {
                case fft_precision_double:
                    std::cerr
                        << "test ordering is incorrect: double precision follows a narrower one"
                        << std::endl;
                    abort();
                    break;
                case fft_precision_single:
                    if(last_cpu_fft_data.precision != fft_precision_double)
                    {
                        std::cerr
                            << "test ordering is incorrect: float precision follows a narrower one"
                            << std::endl;
                        abort();
                    }
                    // convert the input/output to single-precision
                    convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<double, float>(cpu_output.front());
                    });
                    convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                        narrow_precision_inplace<double, float>(cpu_input.front());
                    });
                    break;
                case fft_precision_half:
                    // convert to half precision
                    if(last_cpu_fft_data.precision == fft_precision_double)
                    {
                        convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<double, _Float16>(cpu_output.front());
                        });
                        convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<double, _Float16>(cpu_input.front());
                        });
                    }
                    else if(last_cpu_fft_data.precision == fft_precision_single)
                    {
                        convert_cpu_output_precision = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<float, _Float16>(cpu_output.front());
                        });
                        convert_cpu_input_precision  = std::async(std::launch::async, [&]() {
                            narrow_precision_inplace<float, _Float16>(cpu_input.front());
                        });
                    }
                    else
                    {
                        std::cerr << "unhandled previous precision, cannot convert to half"
                                  << std::endl;
                        abort();
                    }
                    break;
                }
                last_cpu_fft_data.precision = params.precision;
            }
        }
        // If the last result has a smaller batch than the new
        // params, that might be a developer error - tests should be
        // ordered to generate the bigger batch first.  But if tests
        // got filtered or skipped due to insufficient memory, we
        // might never have tried to generate the bigger batch first.
        // So just fall through and redo the CPU FFT.
    }
    else
    {
        // Clear cache explicitly so that even if we didn't get a hit,
        // we're not uselessly holding on to cached cpu input/output
        last_cpu_fft_data = last_cpu_fft_cache();
    }

    // Allocate CPU input
    if(run_fftw)
    {
        cpu_input = allocate_cpu_fft_buffer(
            contiguous_params.precision, contiguous_params.itype, contiguous_params.isize);
    }

    // Create FFTW plan - this may write to input, but that's fine
    // since there's nothing in there right now
    typename fftw_trait<Tfloat>::fftw_plan_type cpu_plan = nullptr;
    if(run_fftw)
    {
        // Normally, we would want to defer allocation of CPU output
        // buffer until when we actually do the CPU FFT.  But if we're
        // using FFTW wisdom, FFTW needs an output buffer at plan
        // creation time.
        if(use_fftw_wisdom)
        {
            cpu_output = allocate_cpu_fft_buffer(
                contiguous_params.precision, contiguous_params.otype, contiguous_params.osize);
        }
        cpu_plan = fftw_plan_via_rocfft<Tfloat>(contiguous_params.length,
                                                contiguous_params.istride,
                                                contiguous_params.ostride,
                                                contiguous_params.nbatch,
                                                contiguous_params.idist,
                                                contiguous_params.odist,
                                                contiguous_params.transform_type,
                                                cpu_input,
                                                cpu_output);

        needed_ram += needed_ram_fftw<Tfloat>(contiguous_params, cpu_plan, verbose);

        if(ramgb > 0 && needed_ram > ramgb * ONE_GiB)
        {
            if(verbose)
            {
                std::cout << "Problem exceeds memory limit; skipped [rocfft_transform]."
                          << std::endl;
            }
            GTEST_SKIP();
            return;
        }
    }

    std::vector<hostbuf> gpu_input_data;

    // allocate and populate the input buffer (cpu/gpu)
    if(run_fftw)
    {
        gpu_input_data = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes_elems);

        //generate the input directly on the gpu
        params.compute_input(ibuffer);

        // Copy the input to CPU
        if(params.itype != contiguous_params.itype || params.istride != contiguous_params.istride
           || params.idist != contiguous_params.idist || params.isize != contiguous_params.isize)
        {
            // Copy input to CPU
            for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
            {
                hip_status = hipMemcpy(gpu_input_data.at(idx).data(),
                                       ibuffer[idx].data(),
                                       ibuffer_sizes[idx],
                                       hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    if(skip_runtime_fails)
                    {
                        GTEST_SKIP() << "hipMemcpy failure with error " << hip_status;
                    }
                    else
                    {
                        GTEST_FAIL() << "hipMemcpy failure with error " << hip_status;
                    }
                }
            }

            copy_buffers(gpu_input_data,
                         cpu_input,
                         params.ilength(),
                         params.nbatch,
                         params.precision,
                         params.itype,
                         params.istride,
                         params.idist,
                         contiguous_params.itype,
                         contiguous_params.istride,
                         contiguous_params.idist,
                         params.ioffset,
                         contiguous_params.ioffset);
        }
        else
        {
            // Copy input to CPU
            for(unsigned int idx = 0; idx < ibuffer.size(); ++idx)
            {
                hip_status = hipMemcpy(cpu_input.at(idx).data(),
                                       ibuffer[idx].data(),
                                       ibuffer_sizes[idx],
                                       hipMemcpyDeviceToHost);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    if(skip_runtime_fails)
                    {
                        GTEST_SKIP() << "hipMemcpy failure with error " << hip_status;
                    }
                    else
                    {
                        GTEST_FAIL() << "hipMemcpy failure with error " << hip_status;
                    }
                }
            }
        }
    }
    else if(fftw_compare)
    {
        gpu_input_data = allocate_host_buffer(params.precision, params.itype, ibuffer_sizes_elems);

        // In case the cached cpu input needed conversion, wait for it
        if(convert_cpu_input_precision.valid())
            convert_cpu_input_precision.get();

        // gets a pre-computed gpu input buffer from the cpu cache
        std::vector<hostbuf>* gpu_input = &cpu_input;

        if(params.itype != contiguous_params.itype || params.istride != contiguous_params.istride
           || params.idist != contiguous_params.idist || params.isize != contiguous_params.isize)
        {
            copy_buffers(cpu_input,
                         gpu_input_data,
                         params.ilength(),
                         params.nbatch,
                         params.precision,
                         contiguous_params.itype,
                         contiguous_params.istride,
                         contiguous_params.idist,
                         params.itype,
                         params.istride,
                         params.idist,
                         {0},
                         params.ioffset);
            gpu_input = &gpu_input_data;
        }

        // Copy input to GPU
        for(unsigned int idx = 0; idx < gpu_input->size(); ++idx)
        {
            hip_status = hipMemcpy(ibuffer[idx].data(),
                                   gpu_input->at(idx).data(),
                                   ibuffer_sizes[idx],
                                   hipMemcpyHostToDevice);

            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                if(skip_runtime_fails)
                {
                    GTEST_SKIP() << "hipMemcpy failure with error " << hip_status;
                }
                else
                {
                    GTEST_FAIL() << "hipMemcpy failure with error " << hip_status;
                }
            }
        }
    }

    if(verbose > 3)
    {
        std::cout << "CPU input:\n";
        contiguous_params.print_ibuffer(cpu_input);
    }

    // compute input norm
    std::shared_future<VectorNorms> cpu_input_norm;
    if(fftw_compare)
        cpu_input_norm = std::async(std::launch::async, [&]() {
            // in case the cached cpu input needed conversion, wait for it
            if(convert_cpu_input_precision.valid())
                convert_cpu_input_precision.get();

            auto input_norm = norm(cpu_input,
                                   contiguous_params.ilength(),
                                   contiguous_params.nbatch,
                                   contiguous_params.precision,
                                   contiguous_params.itype,
                                   contiguous_params.istride,
                                   contiguous_params.idist,
                                   contiguous_params.ioffset);
            if(verbose > 2)
            {
                std::cout << "CPU Input Linf norm:  " << input_norm.l_inf << "\n";
                std::cout << "CPU Input L2 norm:    " << input_norm.l_2 << "\n";
            }
            return input_norm;
        });

    std::vector<gpubuf>  obuffer_data;
    std::vector<gpubuf>* obuffer = &obuffer_data;
    std::vector<void*>   pobuffer;

    // allocate the output buffer

    if(params.placement == fft_placement_inplace)
    {
        obuffer = &ibuffer;
    }
    else
    {
        auto obuffer_sizes = params.obuffer_sizes();
        obuffer_data.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer_data.size(); ++i)
        {
            hip_status = obuffer_data[i].alloc(obuffer_sizes[i]);
            if(hip_status != hipSuccess)
            {
                ++n_hip_failures;
                std::stringstream ss;
                ss << "hipMalloc failure for output buffer " << i << " size " << obuffer_sizes[i]
                   << "(" << bytes_to_GiB(obuffer_sizes[i]) << " GiB)"
                   << " with code " << hipError_to_string(hip_status);
                if(skip_runtime_fails)
                {
                    GTEST_SKIP() << ss.str();
                }
                else
                {
                    GTEST_FAIL() << ss.str();
                }
            }

            // If we're validating output strides, init the
            // output buffer to a known pattern and we can check
            // that the pattern is untouched in places that
            // shouldn't have been touched.
            if(params.check_output_strides)
            {
                hip_status
                    = hipMemset(obuffer_data[i].data(), OUTPUT_INIT_PATTERN, obuffer_sizes[i]);
                if(hip_status != hipSuccess)
                {
                    ++n_hip_failures;
                    if(skip_runtime_fails)
                    {
                        GTEST_SKIP() << "hipMemset failure with error " << hip_status;
                    }
                    else
                    {
                        GTEST_FAIL() << "hipMemset failure with error " << hip_status;
                    }
                }
            }
        }
    }
    pobuffer.resize(obuffer->size());
    for(unsigned int i = 0; i < obuffer->size(); ++i)
    {
        pobuffer[i] = obuffer->at(i).data();
    }

    // Run CPU transform
    //
    // NOTE: This must happen after input is copied to GPU and input
    // norm is computed, since the CPU FFT may overwrite the input.
    VectorNorms              cpu_output_norm;
    std::shared_future<void> cpu_fft;
    if(fftw_compare)
        cpu_fft = std::async(std::launch::async, [&]() {
            // wait for input norm to finish, since we might overwrite input
            cpu_input_norm.get();

            if(run_fftw)
                execute_cpu_fft<Tfloat>(params, contiguous_params, cpu_plan, cpu_input, cpu_output);
            // in case the cached cpu output needed conversion, wait for it
            else if(convert_cpu_output_precision.valid())
                convert_cpu_output_precision.get();

            if(verbose > 3)
            {
                std::cout << "CPU output:\n";
                contiguous_params.print_obuffer(cpu_output);
            }

            cpu_output_norm = norm(cpu_output,
                                   params.olength(),
                                   params.nbatch,
                                   params.precision,
                                   contiguous_params.otype,
                                   contiguous_params.ostride,
                                   contiguous_params.odist,
                                   contiguous_params.ooffset);
            if(verbose > 2)
            {
                std::cout << "CPU Output Linf norm: " << cpu_output_norm.l_inf << "\n";
                std::cout << "CPU Output L2 norm:   " << cpu_output_norm.l_2 << "\n";
            }
        });

    // scatter data out to multi-GPUs if this is a multi-GPU test
    params.multi_gpu_prepare(ibuffer, pibuffer, pobuffer);

    // execute GPU transform
    std::vector<hostbuf> gpu_output
        = allocate_host_buffer(params.precision, params.otype, params.osize);

    execute_gpu_fft(params, pibuffer, pobuffer, *obuffer, gpu_output);

    params.free();

    if(params.check_output_strides)
    {
        check_output_strides<Tparams>(gpu_output, params);
    }

    // compute GPU output norm
    std::shared_future<VectorNorms> gpu_norm;
    if(fftw_compare)
        gpu_norm = std::async(std::launch::async, [&]() {
            return norm(gpu_output,
                        params.olength(),
                        params.nbatch,
                        params.precision,
                        params.otype,
                        params.ostride,
                        params.odist,
                        params.ooffset);
        });

    // compare output
    //
    // Compute the l-infinity and l-2 distance between the CPU and GPU output:
    // wait for cpu FFT so we can compute cutoff

    const auto total_length = std::accumulate(params.length.begin(),
                                              params.length.end(),
                                              static_cast<size_t>(1),
                                              std::multiplies<size_t>());

    std::unique_ptr<std::vector<std::pair<size_t, size_t>>> linf_failures;
    if(verbose > 1)
        linf_failures = std::make_unique<std::vector<std::pair<size_t, size_t>>>();
    double      linf_cutoff;
    VectorNorms diff;

    std::shared_future<void> compare_output;
    if(fftw_compare)
        compare_output = std::async(std::launch::async, [&]() {
            cpu_fft.get();
            linf_cutoff
                = type_epsilon(params.precision) * cpu_output_norm.l_inf * log(total_length);

            diff = distance(cpu_output,
                            gpu_output,
                            params.olength(),
                            params.nbatch,
                            params.precision,
                            contiguous_params.otype,
                            contiguous_params.ostride,
                            contiguous_params.odist,
                            params.otype,
                            params.ostride,
                            params.odist,
                            linf_failures.get(),
                            linf_cutoff,
                            {0},
                            params.ooffset);
        });

    // Update the cache if this current transform is different from
    // what's stored.  But if this transform only has a smaller batch
    // than what's cached, we can still keep the cache around since
    // the input/output we already have is still valid.
    const bool update_last_cpu_fft_data
        = last_cpu_fft_data.length != params.length
          || last_cpu_fft_data.transform_type != params.transform_type
          || last_cpu_fft_data.run_callbacks != params.run_callbacks
          || last_cpu_fft_data.precision != params.precision
          || params.nbatch > last_cpu_fft_data.nbatch;

    // store cpu output in cache
    if(update_last_cpu_fft_data)
    {
        last_cpu_fft_data.length         = params.length;
        last_cpu_fft_data.nbatch         = params.nbatch;
        last_cpu_fft_data.transform_type = params.transform_type;
        last_cpu_fft_data.run_callbacks  = params.run_callbacks;
        last_cpu_fft_data.precision      = params.precision;
    }

    if(compare_output.valid())
        compare_output.get();

    if(!store_to_cache)
        store_to_cache = std::make_unique<StoreCPUDataToCache>(cpu_input, cpu_output);

    Tparams params_inverse;

    if(round_trip)
    {
        params_inverse.inverse_from_forward(params);

        run_round_trip_inverse<Tparams>(
            params_inverse, ibuffer, pobuffer, pibuffer, gpu_input_data);
    }

    if(fftw_compare)
    {
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_2));
        ASSERT_TRUE(std::isfinite(cpu_input_norm.get().l_inf));

        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));
        ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));

        if(verbose > 1)
        {
            std::cout << "GPU output Linf norm: " << gpu_norm.get().l_inf << "\n";
            std::cout << "GPU output L2 norm:   " << gpu_norm.get().l_2 << "\n";
            std::cout << "GPU linf norm failures:";
            std::sort(linf_failures->begin(), linf_failures->end());
            for(const auto& i : *linf_failures)
            {
                std::cout << " (" << i.first << "," << i.second << ")";
            }
            std::cout << std::endl;
        }

        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_inf)) << params.str();
        EXPECT_TRUE(std::isfinite(gpu_norm.get().l_2)) << params.str();
    }

    switch(params.precision)
    {
    case fft_precision_half:
        max_linf_eps_half
            = std::max(max_linf_eps_half, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_half
            = std::max(max_l2_eps_half, diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_single:
        max_linf_eps_single
            = std::max(max_linf_eps_single, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_single = std::max(max_l2_eps_single,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    case fft_precision_double:
        max_linf_eps_double
            = std::max(max_linf_eps_double, diff.l_inf / cpu_output_norm.l_inf / log(total_length));
        max_l2_eps_double = std::max(max_l2_eps_double,
                                     diff.l_2 / cpu_output_norm.l_2 * sqrt(log2(total_length)));
        break;
    }

    if(verbose > 1)
    {
        std::cout << "L2 diff: " << diff.l_2 << "\n";
        std::cout << "Linf diff: " << diff.l_inf << "\n";
    }

    if(fftw_compare)
    {
        EXPECT_TRUE(diff.l_inf <= linf_cutoff)
            << "Linf test failed.  Linf:" << diff.l_inf
            << "\tnormalized Linf: " << diff.l_inf / cpu_output_norm.l_inf
            << "\tcutoff: " << linf_cutoff << params.str();

        EXPECT_TRUE(diff.l_2 / cpu_output_norm.l_2
                    < sqrt(log2(total_length)) * type_epsilon(params.precision))
            << "L2 test failed. L2: " << diff.l_2
            << "\tnormalized L2: " << diff.l_2 / cpu_output_norm.l_2
            << "\tepsilon: " << sqrt(log2(total_length)) * type_epsilon(params.precision)
            << params.str();
    }

    if(round_trip && fftw_compare)
    {
        compare_round_trip_inverse<Tparams>(params_inverse,
                                            contiguous_params,
                                            gpu_input_data,
                                            cpu_input,
                                            cpu_input_norm.get(),
                                            total_length);
    }
}

#endif
