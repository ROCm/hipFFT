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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <sstream>

#include "rider.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

// NB:
// hipFFT-rider keeps the same test interface with rocFFT-rider.
// hipFFT-rider reuses some rocFFT enums to organize test logic internally
// despite test rocFFT or cuFFT underlying.
// By the limitation of hipfftPlanMany, user can not specify arbitrary strides
// for multiple dimensions.

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose;

    // hip Device number for running tests:
    int deviceId;

    // Transform type parameters:
    rocfft_transform_type transformType;
    rocfft_array_type     itype;
    rocfft_array_type     otype;

    // Number of performance trial samples
    int ntrial;

    // Number of batches:
    int nbatch = 1;

    // Scale for transform
    double scale = 1.0;

    // Transform length:
    std::vector<int> length;

    // Transform input and output strides:
    std::vector<int> istride;
    std::vector<int> ostride;

    // Offset to start of buffer (or buffers, for planar format):
    std::vector<int> ioffset;
    std::vector<int> ooffset;

    // Input and output distances:
    int idist;
    int odist;

    // Declare the supported options.

    // clang-format doesn't handle boost program options very well:
    // clang-format off
    po::options_description opdesc("hipfft rider command line options");
    opdesc.add_options()("help,h", "produces this help message")
        ("version,v", "Print queryable version information from the hipfft library")
        ("device", po::value<int>(&deviceId)->default_value(0), "Select a specific device id")
        ("verbose", po::value<int>(&verbose)->default_value(0), "Control output verbosity")
        ("ntrial,N", po::value<int>(&ntrial)->default_value(1), "Trial size for the problem")
        ("notInPlace,o", "Not in-place FFT transform (default: in-place)")
        ("double", "Double precision transform (default: single)")
        ("transformType,t", po::value<rocfft_transform_type>(&transformType)
         ->default_value(rocfft_transform_type_complex_forward),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "idist", po::value<int>(&idist)->default_value(0),
          "input distance between successive members when batch size > 1")
        ( "odist", po::value<int>(&odist)->default_value(0),
          "output distance between successive members when batch size > 1")
        ("scale", po::value<double>(&scale)->default_value(1.0), "Specify the scaling factor ")
        ( "batchSize,b", po::value<int>(&nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<rocfft_array_type>(&itype)
          ->default_value(rocfft_array_type_unset),
          "Array type of input data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ( "otype", po::value<rocfft_array_type>(&otype)
          ->default_value(rocfft_array_type_unset),
          "Array type of output data:\n0) interleaved\n1) planar\n2) real\n3) "
          "hermitian interleaved\n4) hermitian planar")
        ("length",  po::value<std::vector<int>>(&length)->multitoken(), "Lengths.")
        ("istride", po::value<std::vector<int>>(&istride)->multitoken(), "Input strides.")
        ("ostride", po::value<std::vector<int>>(&ostride)->multitoken(), "Output strides.")
        ("ioffset", po::value<std::vector<int>>(&ioffset)->multitoken(), "Input offsets.")
        ("ooffset", po::value<std::vector<int>>(&ooffset)->multitoken(), "Output offsets.");
    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opdesc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << opdesc << std::endl;
        return 0;
    }

    if(vm.count("version"))
    {
        int v;
        hipfftGetVersion(&v);
        std::cout << "version " << v << std::endl;
        return 0;
    }

    if(!vm.count("length"))
    {
        std::cout << "Please specify transform length!" << std::endl;
        std::cout << opdesc << std::endl;
        return 0;
    }

    const rocfft_result_placement place
        = vm.count("notInPlace") ? rocfft_placement_notinplace : rocfft_placement_inplace;
    const rocfft_precision precision
        = vm.count("double") ? rocfft_precision_double : rocfft_precision_single;

    if(vm.count("notInPlace"))
    {
        std::cout << "out-of-place\n";
    }
    else
    {
        std::cout << "in-place\n";
    }

    if(vm.count("ntrial"))
    {
        std::cout << "Running profile with " << ntrial << " samples\n";
    }

    if(vm.count("length"))
    {
        std::cout << "length:";
        for(auto& i : length)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("istride"))
    {
        std::cout << "istride:";
        for(auto& i : istride)
            std::cout << " " << i;
        std::cout << "\n";

        // istride in hipfftPlanMany applies to innermost dimension,
        // which is implicitly to each logic dimension.
        for(auto& i : istride)
        {
            if(i != istride[0])
            {
                std::cerr << "Unspported strides: istride must be identical along all dimensions.";
                return -1;
            }
        }
    }
    if(vm.count("ostride"))
    {
        std::cout << "ostride:";
        for(auto& i : ostride)
            std::cout << " " << i;
        std::cout << "\n";

        for(auto& i : ostride)
        {
            if(i != ostride[0])
            {
                std::cerr << "Unspported strides: ostride must be identical along all dimensions.";
                return -1;
            }
        }
    }

    if(idist > 0)
    {
        std::cout << "idist: " << idist << "\n";
    }
    if(odist > 0)
    {
        std::cout << "odist: " << odist << "\n";
    }

    if(vm.count("ioffset"))
    {
        std::cout << "ioffset:";
        for(auto& i : ioffset)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ooffset"))
    {
        std::cout << "ooffset:";
        for(auto& i : ooffset)
            std::cout << " " << i;
        std::cout << "\n";
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    //HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    // Set default data formats if not yet specified:
    const int dim     = length.size();
    auto      ilength = length;
    if(transformType == rocfft_transform_type_real_inverse)
    {
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    }
    if(istride.size() == 0)
    {
        istride = compute_stride(ilength,
                                 1,
                                 place == rocfft_placement_inplace
                                     && transformType == rocfft_transform_type_real_forward);
    }
    auto olength = length;
    if(transformType == rocfft_transform_type_real_forward)
    {
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    }
    if(ostride.size() == 0)
    {
        ostride = compute_stride(olength,
                                 1,
                                 place == rocfft_placement_inplace
                                     && transformType == rocfft_transform_type_real_inverse);
    }
    check_set_iotypes(place, transformType, itype, otype);
    if(idist == 0)
    {
        idist = set_idist(place, transformType, length, istride);
    }
    if(odist == 0)
    {
        odist = set_odist(place, transformType, length, ostride);
    }

    if(verbose > 0)
    {
        std::cout << "FFT  params:\n";
        std::cout << "\tilength:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tistride:";
        for(auto i : istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tidist: " << idist << std::endl;

        std::cout << "\tolength:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tostride:";
        for(auto i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\todist: " << odist << std::endl;
    }

    hipfftHandle plan;
    LIB_V_THROW(hipfftCreate(&plan), "hipfftCreate failed");

    hipfftType hip_fft_type;
    if(transformType == rocfft_transform_type_complex_forward
       || transformType == rocfft_transform_type_complex_inverse)
    {
        hip_fft_type = (precision == rocfft_precision_single) ? HIPFFT_C2C : HIPFFT_Z2Z;
    }
    else if(transformType == rocfft_transform_type_real_forward)
    {
        hip_fft_type = (precision == rocfft_precision_single) ? HIPFFT_R2C : HIPFFT_D2Z;
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        hip_fft_type = (precision == rocfft_precision_single) ? HIPFFT_C2R : HIPFFT_Z2D;
    }

    int direction = (transformType == rocfft_transform_type_complex_forward
                     || transformType == rocfft_transform_type_real_forward)
                        ? HIPFFT_FORWARD
                        : HIPFFT_BACKWARD;

    int i_stride   = istride[length.size() - 1];
    int o_stride   = ostride[length.size() - 1];
    int inembed[3] = {i_stride * length[0], 0, 0};
    int onembed[3] = {o_stride * length[0], 0, 0};

    if(dim >= 2)
    {
        inembed[1] = i_stride * length[1];
        onembed[1] = o_stride * length[1];
    }
    if(dim >= 3)
    {
        inembed[2] = i_stride * length[2];
        onembed[2] = o_stride * length[2];
    }

    if(transformType == rocfft_transform_type_real_forward)
    {
        switch(dim)
        {
        case 1:
        {
            int n0_complex_elements      = length[0] / 2 + 1;
            int n0_padding_real_elements = n0_complex_elements * 2;
            inembed[0]                   = i_stride * n0_padding_real_elements;
            onembed[0]                   = o_stride * n0_complex_elements;
            break;
        }
        case 2:
        {
            int n1_complex_elements      = length[1] / 2 + 1;
            int n1_padding_real_elements = n1_complex_elements * 2;
            inembed[1]                   = i_stride * n1_padding_real_elements;
            onembed[1]                   = o_stride * n1_complex_elements;
            break;
        }
        case 3:
        {
            int n2_complex_elements      = length[2] / 2 + 1;
            int n2_padding_real_elements = n2_complex_elements * 2;
            inembed[1]                   = i_stride * length[1];
            onembed[1]                   = o_stride * length[1];
            inembed[2]                   = i_stride * n2_padding_real_elements;
            onembed[2]                   = o_stride * n2_complex_elements;
            break;
        }
        }
    }
    else if(transformType == rocfft_transform_type_real_inverse)
    {
        switch(dim)
        {
        case 1:
        {
            int n0_complex_elements      = length[0] / 2 + 1;
            int n0_padding_real_elements = n0_complex_elements * 2;
            onembed[0]                   = o_stride * n0_padding_real_elements;
            inembed[0]                   = i_stride * n0_complex_elements;
            break;
        }
        case 2:
        {
            int n1_complex_elements      = length[1] / 2 + 1;
            int n1_padding_real_elements = n1_complex_elements * 2;
            onembed[1]                   = o_stride * n1_padding_real_elements;
            inembed[1]                   = i_stride * n1_complex_elements;
            break;
        }
        case 3:
        {
            int n2_complex_elements      = length[2] / 2 + 1;
            int n2_padding_real_elements = n2_complex_elements * 2;
            onembed[1]                   = o_stride * length[1];
            inembed[1]                   = i_stride * length[1];
            onembed[2]                   = o_stride * n2_padding_real_elements;
            inembed[2]                   = i_stride * n2_complex_elements;
            break;
        }
        }
    }

    LIB_V_THROW(hipfftPlanMany(&plan,
                               dim,
                               length.data(),
                               inembed,
                               i_stride,
                               idist,
                               onembed,
                               o_stride,
                               odist,
                               hip_fft_type,
                               nbatch),
                "hipfftPlanMany failed");

    // Get work buffer size and allocated associated work buffer is necessary
    void*  work_buf;
    size_t work_buf_size = 0;
    LIB_V_THROW(hipfftGetSizeMany(plan,
                                  dim,
                                  length.data(),
                                  inembed,
                                  i_stride,
                                  idist,
                                  onembed,
                                  o_stride,
                                  odist,
                                  hip_fft_type,
                                  nbatch,
                                  &work_buf_size),
                "hipfftGetSizeMany failed");
    if(work_buf_size)
    {
        HIP_V_THROW(hipMalloc(&work_buf, work_buf_size), "Creating intermediate buffer failed");
        LIB_V_THROW(hipfftSetWorkArea(plan, work_buf), "hipfftSetWorkArea failed");
    }

    // Input data:
    std::vector<size_t> slength, sistride;
    slength.assign(length.begin(), length.end());
    sistride.assign(istride.begin(), istride.end());
    const auto input
        = compute_input(precision, itype, slength, sistride, size_t(idist), size_t(nbatch));

    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        printbuffer(precision, itype, input, ilength, istride, nbatch, idist);
    }

    hipError_t hip_status = hipSuccess;

    // GPU input and output buffers:
    auto               ibuffer_sizes = buffer_sizes(precision, itype, idist, nbatch);
    std::vector<void*> ibuffer(ibuffer_sizes.size());
    for(unsigned int i = 0; i < ibuffer.size(); ++i)
    {
        hip_status = hipMalloc(&ibuffer[i], ibuffer_sizes[i]);
        if(hip_status != hipSuccess)
        {
            std::cerr << "hipMalloc failed!\n";
            exit(1);
        }
    }

    std::vector<void*> obuffer;
    if(place == rocfft_placement_inplace)
    {
        obuffer = ibuffer;
    }
    else
    {
        auto obuffer_sizes = buffer_sizes(precision, otype, odist, nbatch);
        obuffer.resize(obuffer_sizes.size());
        for(unsigned int i = 0; i < obuffer.size(); ++i)
        {
            hip_status = hipMalloc(&obuffer[i], obuffer_sizes[i]);
            if(hip_status != hipSuccess)
            {
                std::cerr << "hipMalloc failed!\n";
                exit(1);
            }
        }
    }

    //Warm up once
    // Copy the input data to the GPU:
    for(int idx = 0; idx < input.size(); ++idx)
    {
        HIP_V_THROW(
            hipMemcpy(ibuffer[idx], input[idx].data(), input[idx].size(), hipMemcpyHostToDevice),
            "hipMemcpy failed");
    }

    switch(hip_fft_type)
    {
    case HIPFFT_R2C:
        hipfftExecR2C(plan, (hipfftReal*)(ibuffer[0]), (hipfftComplex*)(obuffer[0]));
        break;
    case HIPFFT_D2Z:
        hipfftExecD2Z(plan, (hipfftDoubleReal*)(ibuffer[0]), (hipfftDoubleComplex*)(obuffer[0]));
        break;
    case HIPFFT_C2R:
    {
        hipfftExecC2R(plan, (hipfftComplex*)(ibuffer[0]), (hipfftReal*)(obuffer[0]));
        break;
    }
    case HIPFFT_Z2D:
        hipfftExecZ2D(plan, (hipfftDoubleComplex*)(ibuffer[0]), (hipfftDoubleReal*)(obuffer[0]));
        break;
    case HIPFFT_C2C:
        LIB_V_THROW(
            hipfftExecC2C(
                plan, (hipfftComplex*)(ibuffer[0]), (hipfftComplex*)(obuffer[0]), direction),
            "hipfftExecC2C failed");
        break;
    case HIPFFT_Z2Z:
        hipfftExecZ2Z(plan,
                      (hipfftDoubleComplex*)(ibuffer[0]),
                      (hipfftDoubleComplex*)(obuffer[0]),
                      direction);
        break;
    }

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    HIP_V_THROW(hipEventCreate(&start), "hipEventCreate failed");
    HIP_V_THROW(hipEventCreate(&stop), "hipEventCreate failed");
    for(int itrial = 0; itrial < gpu_time.size(); ++itrial)
    {

        // Copy the input data to the GPU:
        for(int idx = 0; idx < input.size(); ++idx)
        {
            HIP_V_THROW(
                hipMemcpy(
                    ibuffer[idx], input[idx].data(), input[idx].size(), hipMemcpyHostToDevice),
                "hipMemcpy failed");
        }

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        switch(hip_fft_type)
        {
        case HIPFFT_R2C:
            hipfftExecR2C(plan, (hipfftReal*)(ibuffer[0]), (hipfftComplex*)(obuffer[0]));
            break;
        case HIPFFT_D2Z:
            hipfftExecD2Z(
                plan, (hipfftDoubleReal*)(ibuffer[0]), (hipfftDoubleComplex*)(obuffer[0]));
            break;
        case HIPFFT_C2R:
        {
            hipfftExecC2R(plan, (hipfftComplex*)(ibuffer[0]), (hipfftReal*)(obuffer[0]));
            break;
        }
        case HIPFFT_Z2D:
            hipfftExecZ2D(
                plan, (hipfftDoubleComplex*)(ibuffer[0]), (hipfftDoubleReal*)(obuffer[0]));
            break;
        case HIPFFT_C2C:
            LIB_V_THROW(
                hipfftExecC2C(
                    plan, (hipfftComplex*)(ibuffer[0]), (hipfftComplex*)(obuffer[0]), direction),
                "hipfftExecC2C failed");
            break;
        case HIPFFT_Z2Z:
            hipfftExecZ2Z(plan,
                          (hipfftDoubleComplex*)(ibuffer[0]),
                          (hipfftDoubleComplex*)(obuffer[0]),
                          direction);
            break;
        }

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        hipEventElapsedTime(&time, start, stop);
        gpu_time[itrial] = time;

        if(verbose > 2)
        {
            auto output = allocate_host_buffer(precision, otype, olength, ostride, odist, nbatch);
            for(int idx = 0; idx < output.size(); ++idx)
            {
                hipMemcpy(
                    output[idx].data(), obuffer[idx], output[idx].size(), hipMemcpyDeviceToHost);
            }
            std::cout << "GPU output:\n";
            printbuffer(precision, otype, output, olength, ostride, nbatch, odist);
        }
    }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;

    std::cout << "Execution gflops:  ";
    const double totsize
        = std::accumulate(length.begin(), length.end(), 1, std::multiplies<size_t>());
    const double k
        = ((itype == rocfft_array_type_real) || (otype == rocfft_array_type_real)) ? 2.5 : 5.0;
    const double opscount = (double)nbatch * k * totsize * log(totsize) / log(2.0);
    for(const auto& i : gpu_time)
    {
        std::cout << " " << opscount / (1e6 * i);
    }
    std::cout << std::endl;

    // Clean up:

    hipfftDestroy(plan);
    if(work_buf_size)
        hipFree(work_buf);
    for(auto& buf : ibuffer)
        hipFree(buf);
    for(auto& buf : obuffer)
        hipFree(buf);

    return 0;
}
