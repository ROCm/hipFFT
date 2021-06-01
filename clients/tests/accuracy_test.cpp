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

#include <cassert>

#include "accuracy_test.h"
#include "rocFFT/shared/gpubuf.h"
#include <boost/scope_exit.hpp>
#include <hipfft.h>

void hipfft_transform(const std::vector<size_t>                                  length,
                      const size_t                                               nbatch,
                      const rocfft_precision                                     precision,
                      const rocfft_transform_type                                transformType,
                      const rocfft_result_placement                              placement,
                      const std::vector<size_t>&                                 cpu_istride,
                      const std::vector<size_t>&                                 cpu_ostride,
                      const size_t                                               cpu_idist,
                      const size_t                                               cpu_odist,
                      const rocfft_array_type                                    cpu_itype,
                      const rocfft_array_type                                    cpu_otype,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_input_copy,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_output,
                      const VectorNorms&                                         cpu_output_norm,
                      std::thread*                                               cpu_output_thread)

{
    const size_t dim = length.size();
    hipfftHandle plan;
    hipfftResult fft_status;
    hipError_t   hip_status;

    rocfft_params gpu_params;
    gpu_params.length         = length;
    gpu_params.precision      = precision;
    gpu_params.placement      = placement;
    gpu_params.transform_type = transformType;
    gpu_params.nbatch         = nbatch;

    gpu_params.istride
        = compute_stride(gpu_params.ilength(),
                         {},
                         gpu_params.placement == rocfft_placement_inplace
                             && gpu_params.transform_type == rocfft_transform_type_real_forward);
    gpu_params.idist = set_idist(
        gpu_params.placement, gpu_params.transform_type, gpu_params.length, gpu_params.istride);
    gpu_params.isize.push_back(gpu_params.idist * gpu_params.nbatch);

    gpu_params.ostride
        = compute_stride(gpu_params.olength(),
                         {},
                         gpu_params.placement == rocfft_placement_inplace
                             && gpu_params.transform_type == rocfft_transform_type_real_inverse);
    gpu_params.odist = set_odist(
        gpu_params.placement, gpu_params.transform_type, gpu_params.length, gpu_params.ostride);
    gpu_params.osize.push_back(gpu_params.odist * gpu_params.nbatch);

    if(transformType == rocfft_transform_type_complex_forward
       || transformType == rocfft_transform_type_complex_inverse)
    {
        gpu_params.itype = rocfft_array_type_complex_interleaved;
        gpu_params.otype = rocfft_array_type_complex_interleaved;
        if(gpu_params.precision == rocfft_precision_single)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_C2C,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_C2C, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_C2C);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_C2C);
        }
        if(gpu_params.precision == rocfft_precision_double)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_Z2Z,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_Z2Z, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_Z2Z);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_Z2Z);
        }
    }
    else if(gpu_params.transform_type == rocfft_transform_type_real_forward)
    {
        gpu_params.itype = rocfft_array_type_real;
        gpu_params.otype = rocfft_array_type_hermitian_interleaved;
        if(gpu_params.precision == rocfft_precision_single)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_R2C,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_R2C, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_R2C);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_R2C);
        }
        if(gpu_params.precision == rocfft_precision_double)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_D2Z,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_D2Z, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_D2Z);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_D2Z);
        }
    }
    else if(gpu_params.transform_type == rocfft_transform_type_real_inverse)
    {
        gpu_params.itype = rocfft_array_type_hermitian_interleaved;
        gpu_params.otype = rocfft_array_type_real;
        if(gpu_params.precision == rocfft_precision_single)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_C2R,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_C2R, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_C2R);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_C2R);
        }
        if(gpu_params.precision == rocfft_precision_double)
        {
            if(gpu_params.nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_params.idist,
                                            nullptr,
                                            1,
                                            gpu_params.odist,
                                            HIPFFT_Z2D,
                                            gpu_params.nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_Z2D, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_Z2D);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_Z2D);
        }
    }
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan creation failure";

    auto gpu_input = allocate_host_buffer<fftwAllocator<char>>(
        gpu_params.precision, gpu_params.itype, {gpu_params.idist * gpu_params.nbatch});

    copy_buffers(cpu_input_copy,
                 gpu_input,
                 gpu_params.ilength(),
                 gpu_params.nbatch,
                 gpu_params.precision,
                 cpu_itype,
                 cpu_istride,
                 cpu_idist,
                 gpu_params.itype,
                 gpu_params.istride,
                 gpu_params.idist,
                 {0},
                 {0});

    gpubuf gpu_output_buffer, gpu_input_buffer;
    hip_status = gpu_input_buffer.alloc(gpu_input[0].size());
    ASSERT_TRUE(hip_status == hipSuccess)
        << "hipMalloc failure for input buffer size " << gpu_input[0].size();
    if(placement == rocfft_placement_notinplace)
    {
        hip_status = gpu_output_buffer.alloc(gpu_params.obuffer_sizes()[0]);
        ASSERT_TRUE(hip_status == hipSuccess)
            << "hipMalloc failure for output buffer size " << gpu_params.obuffer_sizes()[0];
    }

    hip_status = hipMemcpy(
        gpu_input_buffer.data(), gpu_input[0].data(), gpu_input[0].size(), hipMemcpyHostToDevice);
    EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";

    void* ibuf = (void*)gpu_input_buffer.data();
    void* obuf;
    if(placement == rocfft_placement_inplace)
        obuf = ibuf;
    else
        obuf = (void*)gpu_output_buffer.data();

    switch(gpu_params.transform_type)
    {
    case(rocfft_transform_type_complex_forward):
    case(rocfft_transform_type_complex_inverse):
        if(gpu_params.precision == rocfft_precision_single)
            hipfftExecC2C(plan,
                          (hipfftComplex*)ibuf,
                          (hipfftComplex*)obuf,
                          gpu_params.transform_type == rocfft_transform_type_complex_forward
                              ? HIPFFT_FORWARD
                              : HIPFFT_BACKWARD);
        else
            hipfftExecZ2Z(plan,
                          (hipfftDoubleComplex*)ibuf,
                          (hipfftDoubleComplex*)obuf,
                          gpu_params.transform_type == rocfft_transform_type_complex_forward
                              ? HIPFFT_FORWARD
                              : HIPFFT_BACKWARD);
        break;
    case(rocfft_transform_type_real_forward):
        if(gpu_params.precision == rocfft_precision_single)
            hipfftExecR2C(plan, (hipfftReal*)ibuf, (hipfftComplex*)obuf);
        else
            hipfftExecD2Z(plan, (hipfftDoubleReal*)ibuf, (hipfftDoubleComplex*)obuf);
        break;
    case(rocfft_transform_type_real_inverse):
        if(gpu_params.precision == rocfft_precision_single)
            hipfftExecC2R(plan, (hipfftComplex*)ibuf, (hipfftReal*)obuf);
        else
            hipfftExecZ2D(plan, (hipfftDoubleComplex*)ibuf, (hipfftDoubleReal*)obuf);
        break;
    }
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan execution failure";

    auto gpu_output = allocate_host_buffer<fftwAllocator<char>>(
        gpu_params.precision, gpu_params.otype, {gpu_params.odist * gpu_params.nbatch});
    hip_status = hipMemcpy(gpu_output[0].data(), obuf, gpu_output[0].size(), hipMemcpyDeviceToHost);
    EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";

    // std::cout << "gpu_buffer: " << gpu_buffer_size << std::endl;
    // std::cout << "gpu_input:  " << gpu_input[0].size() << std::endl;
    // std::cout << "gpu_output: " << gpu_output[0].size() << std::endl;

    // std::cout << "cpu_input:" << std::endl;
    // printbuffer(precision,
    //             cpu_itype,
    //             cpu_input_copy,
    //             gpu_params.ilength(),
    //             cpu_istride,
    //             nbatch,
    //             cpu_idist,
    //             {0});
    // std::cout << "cpu_output:" << std::endl;
    // printbuffer(gpu_params.precision,
    //             gpu_params.otype,
    //             cpu_output,
    //             gpu_params.olength(),
    //             gpu_params.ostride,
    //             gpu_params.nbatch,
    //             gpu_params.odist,
    //             {0});
    // std::cout << "gpu_input:" << std::endl;
    // printbuffer(gpu_params.precision,
    //             gpu_params.itype,
    //             gpu_input,
    //             gpu_params.ilength(),
    //             gpu_params.istride,
    //             gpu_params.nbatch,
    //             gpu_params.idist,
    //             {0});
    // std::cout << "gpu_output:" << std::endl;
    // printbuffer(gpu_params.precision,
    //             gpu_params.otype,
    //             gpu_output,
    //             gpu_params.olength(),
    //             gpu_params.ostride,
    //             gpu_params.nbatch,
    //             gpu_params.odist,
    //             {0});

    // std::cout << gpu_params.str() << std::endl;

    // Compute the Linf and L2 norm of the GPU output:
    VectorNorms gpu_norm;
    std::thread normthread([&]() {
        gpu_norm = norm(gpu_output,
                        gpu_params.olength(),
                        gpu_params.nbatch,
                        gpu_params.precision,
                        gpu_params.otype,
                        gpu_params.ostride,
                        gpu_params.odist,
                        {0});
    });
    if(cpu_output_thread && cpu_output_thread->joinable())
        cpu_output_thread->join();

    // Compute the Linf and L2 distance between the CPU and GPU output:
    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    const double linf_cutoff = type_epsilon(precision) * cpu_output_norm.l_inf * log(total_length);
    auto         diff        = distance(cpu_output,
                         gpu_output,
                         gpu_params.olength(),
                         gpu_params.nbatch,
                         gpu_params.precision,
                         cpu_otype,
                         cpu_ostride,
                         cpu_odist,
                         gpu_params.otype,
                         gpu_params.ostride,
                         gpu_params.odist,
                         linf_failures,
                         linf_cutoff,
                         {0},
                         {0});
    normthread.join();

    EXPECT_TRUE(std::isfinite(gpu_norm.l_inf)) << gpu_norm.l_inf << " " << cpu_output_norm.l_inf;
    EXPECT_TRUE(std::isfinite(gpu_norm.l_2)) << gpu_norm.l_2 << " " << cpu_output_norm.l_2;
    EXPECT_TRUE(diff.l_2 / cpu_output_norm.l_2 < sqrt(log(total_length)) * type_epsilon(precision))
        << "L_2 diff failure " << diff.l_2 / cpu_output_norm.l_2 << " " << gpu_params.str();

    fft_status = hipfftDestroy(plan);
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan destroy failure";
}

TEST_P(hipfft_accuracy_test, vs_fftw)
{
    rocfft_params params;
    params.length         = std::get<0>(GetParam());
    params.nbatch         = std::get<1>(GetParam());
    params.precision      = std::get<2>(GetParam());
    params.transform_type = std::get<3>(GetParam());
    params.placement      = std::get<4>(GetParam());

    // CPU parameters
    params.istride
        = compute_stride(params.ilength(),
                         params.istride,
                         params.placement == rocfft_placement_inplace
                             && params.transform_type == rocfft_transform_type_real_forward);
    params.itype = contiguous_itype(params.transform_type);
    params.idist
        = set_idist(params.placement, params.transform_type, params.length, params.istride);
    params.isize.push_back(params.idist * params.nbatch);

    params.ostride
        = compute_stride(params.olength(),
                         params.ostride,
                         params.placement == rocfft_placement_inplace
                             && params.transform_type == rocfft_transform_type_real_inverse);
    params.odist
        = set_odist(params.placement, params.transform_type, params.length, params.ostride);
    params.otype = contiguous_otype(params.transform_type);
    params.osize.push_back(params.odist * params.nbatch);

    if(!params.valid(verbose))
    {
        GTEST_SKIP();
    }

    // Generate input
    auto cpu_input      = compute_input<fftwAllocator<char>>(params);
    auto cpu_input_copy = cpu_input; // copy of input (might get overwritten by FFTW).

    // FFTW computation
    decltype(cpu_input) cpu_output;
    VectorNorms         cpu_output_norm;
    std::thread         cpu_output_thread([&]() {
        cpu_output = fftw_via_rocfft(params.length,
                                     params.istride,
                                     params.ostride,
                                     params.nbatch,
                                     params.idist,
                                     params.odist,
                                     params.precision,
                                     params.transform_type,
                                     cpu_input);

        cpu_output_norm = norm(cpu_output,
                               params.olength(),
                               params.nbatch,
                               params.precision,
                               params.otype,
                               params.ostride,
                               params.odist,
                               {0});
    });

    // Clean up threads if transform throws...
    BOOST_SCOPE_EXIT_ALL(&cpu_output_thread)
    {
        if(cpu_output_thread.joinable())
            cpu_output_thread.join();
    };

    // GPU computation and comparison
    hipfft_transform(params.length,
                     params.nbatch,
                     params.precision,
                     params.transform_type,
                     params.placement,
                     params.istride,
                     params.ostride,
                     params.idist,
                     params.odist,
                     params.itype,
                     params.otype,
                     cpu_input_copy,
                     cpu_output,
                     cpu_output_norm,
                     &cpu_output_thread);

    if(cpu_output_thread.joinable())
        cpu_output_thread.join();
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_inf));
    ASSERT_TRUE(std::isfinite(cpu_output_norm.l_2));

    SUCCEED();
}
