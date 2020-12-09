// Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include "accuracy_test.h"
#include "gpubuf.h"
#include <boost/scope_exit.hpp>
#include <hipfft.h>

void hipfft_transform(const std::vector<size_t>                                  length,
                      const size_t                                               nbatch,
                      const rocfft_precision                                     precision,
                      const rocfft_transform_type                                transformType,
                      const std::vector<size_t>&                                 cpu_istride,
                      const std::vector<size_t>&                                 cpu_ostride,
                      const size_t                                               cpu_idist,
                      const size_t                                               cpu_odist,
                      const rocfft_array_type                                    cpu_itype,
                      const rocfft_array_type                                    cpu_otype,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_input_copy,
                      const std::vector<std::vector<char, fftwAllocator<char>>>& cpu_output,
                      const std::pair<double, double>& cpu_output_L2Linfnorm,
                      std::thread*                     cpu_output_thread)

{
    const size_t dim      = length.size();
    const size_t istride0 = 1;
    const size_t ostride0 = 1;
    hipfftHandle plan;
    hipfftResult fft_status;
    hipError_t   hip_status;

    rocfft_array_type itype, otype;

    std::stringstream info;
    info << "\nGPU params:\n";

    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;

    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;

    auto gpu_istride
        = compute_stride(ilength, istride0, transformType == rocfft_transform_type_real_forward);
    auto gpu_ostride
        = compute_stride(olength, ostride0, transformType == rocfft_transform_type_real_inverse);
    const auto gpu_idist = set_idist(rocfft_placement_inplace, transformType, length, gpu_istride);
    const auto gpu_odist = set_odist(rocfft_placement_inplace, transformType, length, gpu_ostride);

    info << "\tilength:";
    for(auto i : ilength)
        info << " " << i;
    info << "\n\tnbatch: " << nbatch;
    info << "\n\tolength:";
    for(auto i : olength)
        info << " " << i;
    info << "\n\tcpu_istride:";
    for(auto i : cpu_istride)
        info << " " << i;
    info << "\n\tcpu_idist: " << cpu_idist;
    info << "\n\tcpu_ostride:";
    for(auto i : cpu_ostride)
        info << " " << i;
    info << "\n\tcpu_odist: " << cpu_odist;
    info << "\n\tgpu_istride:";
    for(auto i : gpu_istride)
        info << " " << i;
    info << "\n\tgpu_idist: " << gpu_idist;
    info << "\n\tgpu_ostride:";
    for(auto i : gpu_ostride)
        info << " " << i;
    info << "\n\tgpu_odist: " << gpu_odist;
    if(precision == rocfft_precision_single)
        info << "\n\tsingle-precision\n";
    else
        info << "\n\tdouble-precision\n";

    if(transformType == rocfft_transform_type_complex_forward
       || transformType == rocfft_transform_type_complex_inverse)
    {
        itype = rocfft_array_type_complex_interleaved;
        otype = rocfft_array_type_complex_interleaved;
        if(precision == rocfft_precision_single)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_C2C,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_C2C, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_C2C);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_C2C);
            info << "\ttype: C2C\n";
        }
        if(precision == rocfft_precision_double)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_Z2Z,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_Z2Z, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_Z2Z);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_Z2Z);
            info << "\ttype: Z2Z\n";
        }
    }
    else if(transformType == rocfft_transform_type_real_forward)
    {
        itype = rocfft_array_type_real;
        otype = rocfft_array_type_hermitian_interleaved;
        if(precision == rocfft_precision_single)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_R2C,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_R2C, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_R2C);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_R2C);
            info << "\ttype: R2C\n";
        }
        if(precision == rocfft_precision_double)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_D2Z,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_D2Z, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_D2Z);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_D2Z);
            info << "\ttype: D2Z\n";
        }
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        itype = rocfft_array_type_hermitian_interleaved;
        otype = rocfft_array_type_real;
        if(precision == rocfft_precision_single)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_C2R,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_C2R, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_C2R);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_C2R);
            info << "\ttype: C2R\n";
        }
        if(precision == rocfft_precision_double)
        {
            if(nbatch > 1)
            {
                int n[3];
                for(int i = 0; i < dim; ++i)
                    n[i] = (int)length[i];
                fft_status = hipfftPlanMany(&plan,
                                            dim,
                                            n,
                                            nullptr,
                                            1,
                                            gpu_idist,
                                            nullptr,
                                            1,
                                            gpu_odist,
                                            HIPFFT_Z2D,
                                            nbatch);
            }
            else if(dim == 1)
                fft_status = hipfftPlan1d(&plan, length[0], HIPFFT_Z2D, 1);
            else if(dim == 2)
                fft_status = hipfftPlan2d(&plan, length[0], length[1], HIPFFT_Z2D);
            else if(dim == 3)
                fft_status = hipfftPlan3d(&plan, length[0], length[1], length[2], HIPFFT_Z2D);
            info << "\ttype: Z2D\n";
        }
    }
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan creation failure";

    auto gpu_input = allocate_host_buffer<fftwAllocator<char>>(
        precision, itype, length, gpu_istride, gpu_idist, nbatch);

    copy_buffers(cpu_input_copy,
                 gpu_input,
                 ilength,
                 nbatch,
                 precision,
                 cpu_itype,
                 cpu_istride,
                 cpu_idist,
                 itype,
                 gpu_istride,
                 gpu_idist);

    auto gpu_buffer_size = buffer_sizes(precision, otype, gpu_idist, nbatch)[0];
    gpu_buffer_size      = std::max(gpu_buffer_size, cpu_input_copy[0].size());

    gpubuf gpu_buffer;
    hip_status = gpu_buffer.alloc(gpu_buffer_size);
    ASSERT_TRUE(hip_status == hipSuccess)
        << "hipMalloc failure for input buffer size " << gpu_buffer_size;

    hip_status = hipMemcpy(
        gpu_buffer.data(), gpu_input[0].data(), gpu_input[0].size(), hipMemcpyHostToDevice);
    EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";

    switch(transformType)
    {
    case(rocfft_transform_type_complex_forward):
    case(rocfft_transform_type_complex_inverse):
        if(precision == rocfft_precision_single)
            hipfftExecC2C(plan,
                          (hipfftComplex*)gpu_buffer.data(),
                          (hipfftComplex*)gpu_buffer.data(),
                          transformType == rocfft_transform_type_complex_forward ? HIPFFT_FORWARD
                                                                                 : HIPFFT_BACKWARD);
        else
            hipfftExecZ2Z(plan,
                          (hipfftDoubleComplex*)gpu_buffer.data(),
                          (hipfftDoubleComplex*)gpu_buffer.data(),
                          transformType == rocfft_transform_type_complex_forward ? HIPFFT_FORWARD
                                                                                 : HIPFFT_BACKWARD);
        break;
    case(rocfft_transform_type_real_forward):
        if(precision == rocfft_precision_single)
            hipfftExecR2C(plan, (hipfftReal*)gpu_buffer.data(), (hipfftComplex*)gpu_buffer.data());
        else
            hipfftExecD2Z(plan,
                          (hipfftDoubleReal*)gpu_buffer.data(),
                          (hipfftDoubleComplex*)gpu_buffer.data());
        break;
    case(rocfft_transform_type_real_inverse):
        if(precision == rocfft_precision_single)
            hipfftExecC2R(plan, (hipfftComplex*)gpu_buffer.data(), (hipfftReal*)gpu_buffer.data());
        else
            hipfftExecZ2D(plan,
                          (hipfftDoubleComplex*)gpu_buffer.data(),
                          (hipfftDoubleReal*)gpu_buffer.data());
        break;
    }
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan execution failure";

    auto gpu_output = allocate_host_buffer<fftwAllocator<char>>(
        precision, otype, olength, gpu_istride, gpu_odist, nbatch);
    hip_status = hipMemcpy(
        gpu_output[0].data(), gpu_buffer.data(), gpu_output[0].size(), hipMemcpyDeviceToHost);
    EXPECT_TRUE(hip_status == hipSuccess) << "hipMemcpy failure";

    info << "\tbuffer size: " << cpu_input_copy[0].size() << " " << gpu_buffer_size << " "
         << gpu_output[0].size() << "\n";

    // Compute the Linf and L2 norm of the GPU output:
    std::pair<double, double> L2LinfnormGPU;
    std::thread               normthread([&]() {
        L2LinfnormGPU
            = LinfL2norm(gpu_output, olength, nbatch, precision, otype, gpu_ostride, gpu_odist);
    });
    if(cpu_output_thread && cpu_output_thread->joinable())
        cpu_output_thread->join();

    // std::cout << info.str() << std::endl;
    // std::cout << "cpu_input:" << std::endl;
    // printbuffer(precision, itype, cpu_input_copy, ilength, cpu_istride, nbatch, cpu_idist);
    // std::cout << "cpu_output:" << std::endl;
    // printbuffer(precision, otype, cpu_output, olength, cpu_ostride, nbatch, cpu_odist);
    // std::cout << "gpu_output:" << std::endl;
    // printbuffer(precision, otype, gpu_output, olength, gpu_ostride, nbatch, gpu_odist);

    // Compute the Linf and L2 distance between the CPU and GPU output:
    std::vector<std::pair<size_t, size_t>> linf_failures;
    const auto                             total_length
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    const double linf_cutoff
        = type_epsilon(precision) * cpu_output_L2Linfnorm.first * log(total_length);
    auto linfl2diff = LinfL2diff(cpu_output,
                                 gpu_output,
                                 olength,
                                 nbatch,
                                 precision,
                                 cpu_otype,
                                 cpu_ostride,
                                 cpu_odist,
                                 otype,
                                 gpu_ostride,
                                 gpu_odist,
                                 linf_failures,
                                 linf_cutoff);
    normthread.join();

    EXPECT_TRUE(std::isfinite(L2LinfnormGPU.first))
        << L2LinfnormGPU.first << " " << cpu_output_L2Linfnorm.first;
    EXPECT_TRUE(std::isfinite(L2LinfnormGPU.second))
        << L2LinfnormGPU.second << " " << cpu_output_L2Linfnorm.second;
    EXPECT_TRUE(linfl2diff.second / cpu_output_L2Linfnorm.second
                < sqrt(log(total_length)) * type_epsilon(precision))
        << "L2 diff failure " << linfl2diff.second / cpu_output_L2Linfnorm.second << " "
        << info.str();

    fft_status = hipfftDestroy(plan);
    EXPECT_TRUE(fft_status == HIPFFT_SUCCESS) << "hipFFT plan destroy failure";
}

TEST_P(hipfft_accuracy_test, vs_fftw)
{
    const std::vector<size_t>   length        = std::get<0>(GetParam());
    const size_t                nbatch        = std::get<1>(GetParam());
    const rocfft_precision      precision     = std::get<2>(GetParam());
    const rocfft_transform_type transformType = std::get<3>(GetParam());

    const size_t dim = length.size();

    // Input cpu parameters
    auto ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    const auto cpu_istride = compute_stride(ilength, 1);
    const auto cpu_itype   = contiguous_itype(transformType);
    const auto cpu_idist
        = set_idist(rocfft_placement_notinplace, transformType, length, cpu_istride);

    // Output cpu parameters
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    const auto cpu_ostride = compute_stride(olength, 1);
    const auto cpu_odist
        = set_odist(rocfft_placement_notinplace, transformType, length, cpu_ostride);
    auto cpu_otype = contiguous_otype(transformType);

    // Generate input
    auto cpu_input = compute_input<fftwAllocator<char>>(
        precision, cpu_itype, length, cpu_istride, cpu_idist, nbatch);
    auto cpu_input_copy = cpu_input; // copy of input (might get overwritten by FFTW).

    // FFTW computation
    decltype(cpu_input)       cpu_output;
    std::pair<double, double> cpu_output_L2Linfnorm;
    std::thread               cpu_output_thread([&]() {
        cpu_output = fftw_via_rocfft(length,
                                     cpu_istride,
                                     cpu_ostride,
                                     nbatch,
                                     cpu_idist,
                                     cpu_odist,
                                     precision,
                                     transformType,
                                     cpu_input);

        cpu_output_L2Linfnorm
            = LinfL2norm(cpu_output, olength, nbatch, precision, cpu_otype, cpu_ostride, cpu_odist);
    });

    // Clean up threads if transform throws...
    BOOST_SCOPE_EXIT_ALL(&cpu_output_thread)
    {
        if(cpu_output_thread.joinable())
            cpu_output_thread.join();
    };

    // GPU computation and comparison
    hipfft_transform(length,
                     nbatch,
                     precision,
                     transformType,
                     cpu_istride,
                     cpu_ostride,
                     cpu_idist,
                     cpu_odist,
                     cpu_itype,
                     cpu_otype,
                     cpu_input_copy,
                     cpu_output,
                     cpu_output_L2Linfnorm,
                     &cpu_output_thread);

    if(cpu_output_thread.joinable())
        cpu_output_thread.join();
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.first));
    ASSERT_TRUE(std::isfinite(cpu_output_L2Linfnorm.second));

    SUCCEED();
}
