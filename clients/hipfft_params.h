// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPFFT_PARAMS_H
#define HIPFFT_PARAMS_H

#include "hipfft.h"
#include "rocFFT/clients/fft_params.h"

inline fft_status fft_status_from_hipfftparams(const hipfftResult_t val)
{
    switch(val)
    {
    case HIPFFT_SUCCESS:
        return fft_status_success;
    case HIPFFT_INVALID_PLAN:
    case HIPFFT_ALLOC_FAILED:
        return fft_status_failure;
    case HIPFFT_INVALID_TYPE:
    case HIPFFT_INVALID_VALUE:
    case HIPFFT_INVALID_SIZE:
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
    case HIPFFT_INVALID_DEVICE:
    case HIPFFT_NOT_IMPLEMENTED:
    case HIPFFT_NOT_SUPPORTED:
        return fft_status_invalid_arg_value;
    case HIPFFT_INTERNAL_ERROR:
    case HIPFFT_EXEC_FAILED:
    case HIPFFT_SETUP_FAILED:
    case HIPFFT_UNALIGNED_DATA:
    case HIPFFT_PARSE_ERROR:
        return fft_status_failure;
    case HIPFFT_NO_WORKSPACE:
        return fft_status_invalid_work_buffer;
    default:
        return fft_status_failure;
    }
}

inline std::string hipfftResult_string(const hipfftResult_t val)
{
    switch(val)
    {
    case HIPFFT_SUCCESS:
        return "HIPFFT_SUCCESS (0)";
    case HIPFFT_INVALID_PLAN:
        return "HIPFFT_INVALID_PLAN (1)";
    case HIPFFT_ALLOC_FAILED:
        return "HIPFFT_ALLOC_FAILED (2)";
    case HIPFFT_INVALID_TYPE:
        return "HIPFFT_INVALID_TYPE (3)";
    case HIPFFT_INVALID_VALUE:
        return "HIPFFT_INVALID_VALUE (4)";
    case HIPFFT_INTERNAL_ERROR:
        return "HIPFFT_INTERNAL_ERROR (5)";
    case HIPFFT_EXEC_FAILED:
        return "HIPFFT_EXEC_FAILED (6)";
    case HIPFFT_SETUP_FAILED:
        return "HIPFFT_SETUP_FAILED (7)";
    case HIPFFT_INVALID_SIZE:
        return "HIPFFT_INVALID_SIZE (8)";
    case HIPFFT_UNALIGNED_DATA:
        return "HIPFFT_UNALIGNED_DATA (9)";
    case HIPFFT_INCOMPLETE_PARAMETER_LIST:
        return "HIPFFT_INCOMPLETE_PARAMETER_LIST (10)";
    case HIPFFT_INVALID_DEVICE:
        return "HIPFFT_INVALID_DEVICE (11)";
    case HIPFFT_PARSE_ERROR:
        return "HIPFFT_PARSE_ERROR (12)";
    case HIPFFT_NO_WORKSPACE:
        return "HIPFFT_NO_WORKSPACE (13)";
    case HIPFFT_NOT_IMPLEMENTED:
        return "HIPFFT_NOT_IMPLEMENTED (14)";
    case HIPFFT_NOT_SUPPORTED:
        return "HIPFFT_NOT_SUPPORTED (16)";
    default:
        return "invalid hipfftResult";
    }
}

class hipfft_params : public fft_params
{
public:
    // plan handles are pointers for rocFFT backend, and ints for cuFFT
#ifdef __HIP_PLATFORM_AMD__
    static constexpr hipfftHandle INVALID_PLAN_HANDLE = nullptr;
#else
    static constexpr hipfftHandle INVALID_PLAN_HANDLE = -1;
#endif

    hipfftHandle plan = INVALID_PLAN_HANDLE;

    hipfftType hipfft_transform_type;
    int        direction;

    std::vector<int> int_length;
    std::vector<int> int_inembed;
    std::vector<int> int_onembed;

    std::vector<long long int> ll_length;
    std::vector<long long int> ll_inembed;
    std::vector<long long int> ll_onembed;

    hipfft_params(){};

    hipfft_params(const fft_params& p)
        : fft_params(p){};

    ~hipfft_params()
    {
        free();
    };

    void free()
    {
        if(plan != INVALID_PLAN_HANDLE)
        {
            hipfftDestroy(plan);
            plan = INVALID_PLAN_HANDLE;
        }
    }

    size_t vram_footprint() override
    {
        size_t val = fft_params::vram_footprint();
        if(setup_structs() != fft_status_success)
        {
            throw std::runtime_error("Struct setup failed");
        }

        workbuffersize = 0;

        // Hack for estimating buffer requirements.
        workbuffersize = 3 * val;

        val += workbuffersize;
        return val;
    }

    fft_status setup_structs()
    {
        switch(transform_type)
        {
        case 0:
            hipfft_transform_type = (precision == fft_precision_single) ? HIPFFT_C2C : HIPFFT_Z2Z;
            direction             = HIPFFT_FORWARD;
            break;
        case 1:
            hipfft_transform_type = (precision == fft_precision_single) ? HIPFFT_C2C : HIPFFT_Z2Z;
            direction             = HIPFFT_BACKWARD;

            break;
        case 2:
            hipfft_transform_type = (precision == fft_precision_single) ? HIPFFT_R2C : HIPFFT_D2Z;
            direction             = HIPFFT_FORWARD;
            break;
        case 3:
            hipfft_transform_type = (precision == fft_precision_single) ? HIPFFT_C2R : HIPFFT_Z2D;
            direction             = HIPFFT_BACKWARD;
            break;
        default:
            throw std::runtime_error("Invalid transform type");
        }

        int_length.resize(dim());
        int_inembed.resize(dim());
        int_onembed.resize(dim());

        ll_length.resize(dim());
        ll_inembed.resize(dim());
        ll_onembed.resize(dim());
        switch(dim())
        {
        case 3:
            ll_inembed[2] = istride[1] / istride[2];
            ll_onembed[2] = ostride[1] / ostride[2];
            [[fallthrough]];
        case 2:
            ll_inembed[1] = istride[0] / istride[1];
            ll_onembed[1] = ostride[0] / ostride[1];
            [[fallthrough]];
        case 1:
            ll_inembed[0] = istride[dim() - 1];
            ll_onembed[0] = ostride[dim() - 1];
            break;
        default:
            throw std::runtime_error("Invalid dimension");
        }

        for(size_t i = 0; i < dim(); ++i)
        {
            ll_length[i]   = length[i];
            int_length[i]  = length[i];
            int_inembed[i] = ll_inembed[i];
            int_onembed[i] = ll_onembed[i];
        }

        hipfftResult ret = HIPFFT_SUCCESS;
        return fft_status_from_hipfftparams(ret);
    }

    fft_status create_plan() override
    {
        auto fft_ret = setup_structs();
        if(fft_ret != fft_status_success)
        {
            return fft_ret;
        }

        hipfftResult ret{HIPFFT_EXEC_FAILED};

        if(plan == INVALID_PLAN_HANDLE)
        {
            ret = hipfftCreate(&plan);
            if(ret != HIPFFT_SUCCESS)
            {
                std::stringstream ss;
                ss << "hipfftCreate failed with code ";
                ss << hipfftResult_string(ret);
                throw std::runtime_error(ss.str());
            }
            if(scale_factor != 1.0)
            {
                ret = hipfftExtPlanScaleFactor(plan, scale_factor);
                if(ret != HIPFFT_SUCCESS)
                {
                    std::stringstream ss;
                    ss << "hipfftExtPlanScaleFactor failed with code ";
                    ss << hipfftResult_string(ret);
                    throw std::runtime_error(ss.str());
                }
            }

#if 1
            try
            {
                ret = hipfftMakePlanMany(plan,
                                         dim(),
                                         int_length.data(),
                                         int_inembed.data(),
                                         istride[dim() - 1],
                                         idist,
                                         int_onembed.data(),
                                         ostride[dim() - 1],
                                         odist,
                                         hipfft_transform_type,
                                         nbatch,
                                         &workbuffersize);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what();
            }
            catch(...)
            {
                std::cerr << "unkown exception in hipfftPlanMany" << std::endl;
            }
            if(ret != HIPFFT_SUCCESS)
            {
                throw std::runtime_error("hipfftPlanMany failed");
            }
#else
            // TODO: enable when implemented in hipFFT for rocFFT.
            try
            {
                ret = hipfftMakePlanMany64(plan,
                                           dim(),
                                           ll_length.data(),
                                           ll_inembed.data(),
                                           istride[dim() - 1],
                                           idist,
                                           ll_onembed.data(),
                                           ostride[dim() - 1],
                                           odist,
                                           hipfft_transform_type,
                                           nbatch,
                                           &workbuffersize);
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what();
            }
            catch(...)
            {
                std::cerr << "unkown exception in hipfftPlanMany64" << std::endl;
            }
            if(ret != HIPFFT_SUCCESS)
            {
                std::stringstream ss;
                ss << "hipfftMakePlanMany64 failed with code ";
                ss << hipfftResult_string(ret);
                throw std::runtime_error(ss.str());
            }
#endif
        }

        fft_ret = fft_status_from_hipfftparams(ret);

        if(fft_ret != fft_status_success)
        {
            return fft_ret;
        }

        return fft_status_success;
    }

    virtual fft_status execute(void** in, void** out) override
    {
        return execute(in[0], out[0]);
    };

    fft_status execute(void* ibuffer, void* obuffer)
    {
        hipfftResult ret{HIPFFT_EXEC_FAILED};
        try
        {
            switch(hipfft_transform_type)
            {
            case HIPFFT_R2C:
                ret = hipfftExecR2C(
                    plan,
                    (hipfftReal*)ibuffer,
                    (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                break;
            case HIPFFT_D2Z:
                ret = hipfftExecD2Z(
                    plan,
                    (hipfftDoubleReal*)ibuffer,
                    (hipfftDoubleComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                break;
            case HIPFFT_C2R:
                ret = hipfftExecC2R(
                    plan,
                    (hipfftComplex*)ibuffer,
                    (hipfftReal*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                break;
            case HIPFFT_Z2D:
                ret = hipfftExecZ2D(
                    plan,
                    (hipfftDoubleComplex*)ibuffer,
                    (hipfftDoubleReal*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                break;
            case HIPFFT_C2C:
                ret = hipfftExecC2C(
                    plan,
                    (hipfftComplex*)ibuffer,
                    (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer),
                    direction);
                break;
            case HIPFFT_Z2Z:
                ret = hipfftExecZ2Z(
                    plan,
                    (hipfftDoubleComplex*)ibuffer,
                    (hipfftDoubleComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer),
                    direction);
                break;
            default:
                throw std::runtime_error("Invalid execution type");
            }
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            std::cerr << "unkown exception in execute(void* ibuffer, void* obuffer)" << std::endl;
        }
        return fft_status_from_hipfftparams(ret);
    }
};

#endif
