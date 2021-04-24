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

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>

#include "rider.h"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int input_buffer_size(hipfftType type, int dist, int nbatch)
{
    switch(type)
    {
    case HIPFFT_Z2D:
    case HIPFFT_Z2Z:
        return dist * nbatch * 16;
    case HIPFFT_D2Z:
        return dist * nbatch * 8;
    case HIPFFT_C2R:
    case HIPFFT_C2C:
        return dist * nbatch * 8;
    case HIPFFT_R2C:
        return dist * nbatch * 4;
    }
}

int output_buffer_size(hipfftType type, int dist, int nbatch)
{
    switch(type)
    {
    case HIPFFT_D2Z:
    case HIPFFT_Z2Z:
        return dist * nbatch * 16;
    case HIPFFT_Z2D:
        return dist * nbatch * 8;
    case HIPFFT_R2C:
    case HIPFFT_C2C:
        return dist * nbatch * 8;
    case HIPFFT_C2R:
        return dist * nbatch * 4;
    }
}

std::vector<char> compute_input(hipfftType type, int dist, int nbatch)
{
    auto              nbytes = input_buffer_size(type, dist, nbatch);
    std::vector<char> buffer(nbytes);

    std::random_device rd;
    std::mt19937       gen(rd());

    switch(type)
    {
    case HIPFFT_Z2D:
    case HIPFFT_Z2Z:
    {
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto x = (std::complex<double>*)buffer.data();
#pragma omp parallel for
        for(size_t i = 0; i < dist * nbatch; ++i)
        {
            x[i].real(dis(gen));
            x[i].imag(dis(gen));
        }
    }
    break;
    case HIPFFT_D2Z:
    {
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        double* x = (double*)buffer.data();
#pragma omp parallel for
        for(size_t i = 0; i < dist * nbatch; ++i)
        {
            x[i] = dis(gen);
        }
    }
    break;
    case HIPFFT_C2R:
    case HIPFFT_C2C:
    {
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        auto x = (std::complex<float>*)buffer.data();
#pragma omp parallel for
        for(size_t i = 0; i < dist * nbatch; ++i)
        {
            x[i].real(dis(gen));
            x[i].imag(dis(gen));
        }
    }
    break;
    case HIPFFT_R2C:
    {
        std::uniform_real_distribution<float> dis(0.0, 1.0);

        float* x = (float*)buffer.data();
#pragma omp parallel for
        for(size_t i = 0; i < dist * nbatch; ++i)
        {
            x[i] = dis(gen);
        }
    }
    break;
    }

    return buffer;
}

template <typename T1, typename T2>
bool increment_rowmajor(std::vector<T1>& index, const std::vector<T2>& length)
{
    for(int idim = length.size(); idim-- > 0;)
    {
        if(index[idim] < length[idim])
        {
            if(++index[idim] == length[idim])
            {
                index[idim] = 0;
                continue;
            }
            // we know we were able to increment something and didn't hit the end
            return true;
        }
    }
    // End the loop when we get back to the start:
    return !std::all_of(index.begin(), index.end(), [](int i) { return i == 0; });
}

template <typename Toutput, typename Tstream = std::ostream>
inline void printbuffer(const Toutput*          output,
                        const std::vector<int>& length,
                        const std::vector<int>& stride,
                        const int               nbatch,
                        const int               dist,
                        const int               offset,
                        Tstream&                stream = std::cout)
{
    auto i_base = 0;
    for(auto b = 0; b < nbatch; b++, i_base += dist)
    {
        std::vector<int> index(length.size());
        std::fill(index.begin(), index.end(), 0);
        do
        {
            const int i
                = std::inner_product(index.begin(), index.end(), stride.begin(), i_base + offset);
            stream << output[i] << " ";
            for(int i = index.size(); i-- > 0;)
            {
                if(index[i] == (length[i] - 1))
                {
                    stream << "\n";
                }
                else
                {
                    break;
                }
            }
        } while(increment_rowmajor(index, length));
        stream << std::endl;
    }
}

int main(int argc, char* argv[])
{
    // This helps with mixing output of both wide and narrow characters to the screen
    std::ios::sync_with_stdio(false);

    // Control output verbosity:
    int verbose;

    // hip Device number for running tests:
    int deviceId;

    // Transform type parameters:
    int        itransformType;
    hipfftType transformType;

    // Number of performance trial samples
    int ntrial;

    // Number of batches:
    int nbatch = 1;

    // Transform length:
    std::vector<int> length;

    // Transform input and output strides:
    std::vector<int> istride;
    std::vector<int> ostride;

    // Offset to start of buffer:
    std::vector<int> ioffset;
    std::vector<int> ooffset;

    // Input and output distances:
    int idist;
    int odist;

    int itype;
    int otype; // Ignored; for compatibility with rocfft-rider.
    
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
        ("transformType,t", po::value<int>(&itransformType)
         ->default_value(0),
         "Type of transform:\n0) complex forward\n1) complex inverse\n2) real "
         "forward\n3) real inverse")
        ( "idist", po::value<int>(&idist)->default_value(0),
          "input distance between successive members when batch size > 1")
        ( "odist", po::value<int>(&odist)->default_value(0),
          "output distance between successive members when batch size > 1")
        ( "batchSize,b", po::value<int>(&nbatch)->default_value(1),
          "If this value is greater than one, arrays will be used ")
        ( "itype", po::value<int>(&itype))
        ( "otype", po::value<int>(&otype))
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

    const bool inplace          = !bool(vm.count("notInPlace"));
    const bool double_precision = bool(vm.count("double"));
    int        direction;
    switch(itransformType)
    {
    case 0:
        transformType = double_precision ? HIPFFT_Z2Z : HIPFFT_C2C;
        direction     = HIPFFT_FORWARD;
        break;
    case 1:
        transformType = double_precision ? HIPFFT_Z2Z : HIPFFT_C2C;
        direction     = HIPFFT_BACKWARD;
        // backward
        break;
    case 2:
        transformType = double_precision ? HIPFFT_D2Z : HIPFFT_R2C;
        direction     = HIPFFT_FORWARD;
        break;
    case 3:
        transformType = double_precision ? HIPFFT_Z2D : HIPFFT_C2R;
        direction     = HIPFFT_BACKWARD;
        break;
    }

    std::cout << "Placement: ";
    if(!inplace)
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
        std::cout << "Length:";
        for(auto& i : length)
            std::cout << " " << i;
        std::cout << "\n";
    }

    if(vm.count("istride"))
    {
        std::cout << "Input stride:";
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
        std::cout << "Output stride:";
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
        std::cout << "Input distance: " << idist << "\n";
    }
    if(odist > 0)
    {
        std::cout << "Output distance: " << odist << "\n";
    }

    if(vm.count("ioffset"))
    {
        std::cout << "Input offset:";
        for(auto& i : ioffset)
            std::cout << " " << i;
        std::cout << "\n";
    }
    if(vm.count("ooffset"))
    {
        std::cout << "Output offset:";
        for(auto& i : ooffset)
            std::cout << " " << i;
        std::cout << "\n";
    }

    std::cout << std::flush;

    // Fixme: set the device id properly after the IDs are synced
    // bewteen hip runtime and rocm-smi.
    //HIP_V_THROW(hipSetDevice(deviceId), "set device failed!");

    // Set default data formats if not yet specified:
    const int dim = length.size();

    auto ilength = length;
    if(transformType == HIPFFT_C2R || transformType == HIPFFT_Z2D)
    {
        ilength[dim - 1] = ilength[dim - 1] / 2 + 1;
    }
    if(istride.empty())
    {
        istride = compute_stride(
            ilength, {1}, inplace && (transformType == HIPFFT_R2C || transformType == HIPFFT_D2Z));
    }

    auto olength = length;
    if(transformType == HIPFFT_R2C || transformType == HIPFFT_D2Z)
    {
        olength[dim - 1] = olength[dim - 1] / 2 + 1;
    }
    if(ostride.empty())
    {
        ostride = compute_stride(
            olength, {1}, inplace && (transformType == HIPFFT_C2R || transformType == HIPFFT_Z2D));
    }

    if(idist == 0)
    {
        idist = std::accumulate(ilength.cbegin(), ilength.cend(), 1, std::multiplies<size_t>());
    }
    if(odist == 0)
    {
        odist = std::accumulate(olength.cbegin(), olength.cend(), 1, std::multiplies<size_t>());
    }

    if(verbose > 0)
    {
        std::cout << "FFT  params:\n";
        std::cout << "\tInput length:";
        for(auto i : ilength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tInput stride:";
        for(auto i : istride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tInput distance: " << idist << std::endl;

        std::cout << "\tOutput length:";
        for(auto i : olength)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tOutput stride:";
        for(auto i : ostride)
            std::cout << " " << i;
        std::cout << "\n";
        std::cout << "\tOutput distance: " << odist << std::endl;
    }

    hipfftHandle plan;
    LIB_V_THROW(hipfftCreate(&plan), "hipfftCreate failed");

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

    if(transformType == HIPFFT_R2C || transformType == HIPFFT_D2Z)
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
    else if(transformType == HIPFFT_C2R || transformType == HIPFFT_Z2D)
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
                               transformType,
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
                                  transformType,
                                  nbatch,
                                  &work_buf_size),
                "hipfftGetSizeMany failed");
    if(work_buf_size)
    {
        HIP_V_THROW(hipMalloc(&work_buf, work_buf_size), "Creating intermediate buffer failed");
        LIB_V_THROW(hipfftSetWorkArea(plan, work_buf), "hipfftSetWorkArea failed");
    }

    hipError_t hip_status = hipSuccess;

    // GPU input and output buffers:
    auto ibuffer_size = input_buffer_size(transformType, idist, nbatch);
    auto obuffer_size = output_buffer_size(transformType, odist, nbatch);

    void* ibuffer;
    HIP_V_THROW(hipMalloc(&ibuffer, ibuffer_size), "hipMalloc failed");

    void* obuffer;
    if(inplace)
    {
        obuffer = ibuffer;
    }
    else
    {
        HIP_V_THROW(hipMalloc(&obuffer, obuffer_size), "hipMalloc failed");
    }

    // Input data:
    const auto input = compute_input(transformType, idist, nbatch);

    if(verbose > 1)
    {
        std::cout << "GPU input:\n";
        switch(transformType)
        {
        case HIPFFT_Z2D:
        case HIPFFT_Z2Z:
            printbuffer((std::complex<double>*)input.data(), ilength, istride, nbatch, idist, 0);
            break;
        case HIPFFT_D2Z:
            printbuffer((double*)input.data(), ilength, istride, nbatch, idist, 0);
            break;
        case HIPFFT_C2R:
        case HIPFFT_C2C:
            printbuffer((std::complex<float>*)input.data(), ilength, istride, nbatch, idist, 0);
            break;
        case HIPFFT_R2C:
            printbuffer((float*)input.data(), ilength, istride, nbatch, idist, 0);
            break;
        }
    }

    // Copy the input data to the GPU:
    HIP_V_THROW(hipMemcpy(ibuffer, input.data(), ibuffer_size, hipMemcpyHostToDevice),
                "hipMemcpy failed");

    // Run the transform several times and record the execution time:
    std::vector<double> gpu_time(ntrial);

    hipEvent_t start, stop;
    HIP_V_THROW(hipEventCreate(&start), "hipEventCreate failed");
    HIP_V_THROW(hipEventCreate(&stop), "hipEventCreate failed");

    // Warm up once (itrial == -1 corresponds to the warm-up trial)
    for(int itrial = -1; itrial < int(gpu_time.size()); ++itrial)
    {
        // Copy the input data to the GPU:
        HIP_V_THROW(hipMemcpy(ibuffer, input.data(), input.size(), hipMemcpyHostToDevice),
                    "hipMemcpy failed");

        HIP_V_THROW(hipEventRecord(start), "hipEventRecord failed");

        switch(transformType)
        {
        case HIPFFT_R2C:
            LIB_V_THROW(hipfftExecR2C(plan, (hipfftReal*)(ibuffer), (hipfftComplex*)(obuffer)),
                        "hipfftExecR2C failed");
            break;
        case HIPFFT_D2Z:
            LIB_V_THROW(
                hipfftExecD2Z(plan, (hipfftDoubleReal*)(ibuffer), (hipfftDoubleComplex*)(obuffer)),
                "hipfftExecD2Z failed");
            break;
        case HIPFFT_C2R:
            LIB_V_THROW(hipfftExecC2R(plan, (hipfftComplex*)(ibuffer), (hipfftReal*)(obuffer)),
                        "hipfftExecC2R failed");
            break;
        case HIPFFT_Z2D:
            LIB_V_THROW(
                hipfftExecZ2D(plan, (hipfftDoubleComplex*)(ibuffer), (hipfftDoubleReal*)(obuffer)),
                "hipfftExecZ2D failed");
            break;
        case HIPFFT_C2C:
            LIB_V_THROW(hipfftExecC2C(
                            plan, (hipfftComplex*)(ibuffer), (hipfftComplex*)(obuffer), direction),
                        "hipfftExecC2C failed");
            break;
        case HIPFFT_Z2Z:
            LIB_V_THROW(hipfftExecZ2Z(plan,
                                      (hipfftDoubleComplex*)(ibuffer),
                                      (hipfftDoubleComplex*)(obuffer),
                                      direction),
                        "hipfftExecZ2Z failed");
            break;
        }

        HIP_V_THROW(hipEventRecord(stop), "hipEventRecord failed");
        HIP_V_THROW(hipEventSynchronize(stop), "hipEventSynchronize failed");

        float time;
        hipEventElapsedTime(&time, start, stop);
        if(itrial > -1)
        {
            gpu_time[itrial] = time;
        }

        if(verbose > 1)
        {
            std::vector<char> output(obuffer_size);
            HIP_V_THROW(hipMemcpy(output.data(), obuffer, output.size(), hipMemcpyDeviceToHost),
                        "hipMemcpy failed");

            std::cout << "GPU output:\n";
            switch(transformType)
            {
            case HIPFFT_D2Z:
            case HIPFFT_Z2Z:
                printbuffer(
                    (std::complex<double>*)output.data(), olength, ostride, nbatch, odist, 0);
                break;
            case HIPFFT_Z2D:
                printbuffer((double*)output.data(), olength, ostride, nbatch, odist, 0);
                break;
            case HIPFFT_R2C:
            case HIPFFT_C2C:
                printbuffer(
                    (std::complex<float>*)output.data(), olength, ostride, nbatch, odist, 0);
                break;
            case HIPFFT_C2R:
                printbuffer((float*)output.data(), olength, ostride, nbatch, odist, 0);
                break;
            }
        }
    }

    std::cout << "\nExecution gpu time:";
    for(const auto& i : gpu_time)
    {
        std::cout << " " << i;
    }
    std::cout << " ms" << std::endl;

    // Clean up:

    hipfftDestroy(plan);
    if(work_buf_size)
        hipFree(work_buf);
    hipFree(ibuffer);
    if(!inplace)
        hipFree(obuffer);

    return 0;
}
