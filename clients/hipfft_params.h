// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <optional>

#include "../shared/fft_params.h"
#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"

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

    // hipFFT has two ways to specify transform type - the hipfftType
    // enum, and separate hipDataType enums for input/output.
    // hipfftType has no way to express an fp16 transform, so
    // hipfft_transform_type will not be set in that case.
    std::optional<hipfftType> hipfft_transform_type;
    hipDataType               inputType  = HIP_C_32F;
    hipDataType               outputType = HIP_C_32F;

    int direction;

    std::vector<int> int_length;
    std::vector<int> int_inembed;
    std::vector<int> int_onembed;

    std::vector<long long int> ll_length;
    std::vector<long long int> ll_inembed;
    std::vector<long long int> ll_onembed;

    struct hipLibXtDesc_deleter
    {
        void operator()(hipLibXtDesc* d)
        {
            hipfftXtFree(d);
        }
    };
    // allocated memory on devices for multi-GPU transforms - inplace
    // just uses xt_output
    std::unique_ptr<hipLibXtDesc, hipLibXtDesc_deleter> xt_input;
    std::unique_ptr<hipLibXtDesc, hipLibXtDesc_deleter> xt_output;

    // backend library can write N worksize values for N GPUs, so
    // allocate a vector for that if necessary
    std::vector<size_t> xt_worksize;

    // pointer we pass to the backend library.  By default point to the
    // single-GPU workbuffer size.
    size_t* workbuffersize_ptr;

    hipfft_params()
    {
        workbuffersize_ptr = &workbuffersize;
    }

    hipfft_params(const fft_params& p)
        : fft_params(p)
    {
        workbuffersize_ptr = &workbuffersize;
    }

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
        xt_input.reset();
        xt_output.reset();
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
        // set direction
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_real_forward:
            direction = HIPFFT_FORWARD;
            break;
        case fft_transform_type_complex_inverse:
        case fft_transform_type_real_inverse:
            direction = HIPFFT_BACKWARD;
            break;
        }

        // set i/o types and transform type
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_C_16F;
                outputType = HIP_C_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_C_32F;
                outputType            = HIP_C_32F;
                hipfft_transform_type = HIPFFT_C2C;
                break;
            case fft_precision_double:
                inputType             = HIP_C_64F;
                outputType            = HIP_C_64F;
                hipfft_transform_type = HIPFFT_Z2Z;
                break;
            }
            break;
        }
        case fft_transform_type_real_forward:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_R_16F;
                outputType = HIP_C_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_R_32F;
                outputType            = HIP_C_32F;
                hipfft_transform_type = HIPFFT_R2C;
                break;
            case fft_precision_double:
                inputType             = HIP_R_64F;
                outputType            = HIP_C_64F;
                hipfft_transform_type = HIPFFT_D2Z;
                break;
            }
            break;
        }
        case fft_transform_type_real_inverse:
        {
            switch(precision)
            {
            case fft_precision_half:
                inputType  = HIP_C_16F;
                outputType = HIP_R_16F;
                hipfft_transform_type.reset();
                break;
            case fft_precision_single:
                inputType             = HIP_C_32F;
                outputType            = HIP_R_32F;
                hipfft_transform_type = HIPFFT_C2R;
                break;
            case fft_precision_double:
                inputType             = HIP_C_64F;
                outputType            = HIP_R_64F;
                hipfft_transform_type = HIPFFT_Z2D;
                break;
            }
            break;
        }
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

        hipfftResult ret{HIPFFT_INTERNAL_ERROR};
        switch(get_create_type())
        {
        case PLAN_Nd:
        {
            ret = create_plan_Nd();
            break;
        }
        case PLAN_MANY:
        {
            ret = create_plan_many();
            break;
        }
        case CREATE_MAKE_PLAN_Nd:
        {
            ret = create_make_plan_Nd();
            break;
        }
        case CREATE_MAKE_PLAN_MANY:
        {
            ret = create_make_plan_many();
            break;
        }
        case CREATE_MAKE_PLAN_MANY64:
        {
            ret = create_make_plan_many64();
            break;
        }
        case CREATE_XT_MAKE_PLAN_MANY:
        {
            ret = create_xt_make_plan_many();
            break;
        }
        default:
        {
            throw std::runtime_error("no valid plan creation type");
        }
        }

        return fft_status_from_hipfftparams(ret);
    }

    void validate_fields() const override
    {
        // hipFFT interprets any present field info as "use hipfftXt
        // APIs to distribute data to multiple GPUs".  Since the
        // library comes up with the data distribution, there's nothing
        // to validate on the client side.
    }

    fft_status set_callbacks(void* load_cb_host,
                             void* load_cb_data,
                             void* store_cb_host,
                             void* store_cb_data) override
    {
        if(run_callbacks)
        {
            if(!hipfft_transform_type)
                throw std::runtime_error("callbacks require a valid hipfftType");

            hipfftResult ret{HIPFFT_EXEC_FAILED};
            switch(*hipfft_transform_type)
            {
            case HIPFFT_R2C:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_REAL, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_D2Z:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_REAL_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_C2R:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(plan, &store_cb_host, HIPFFT_CB_ST_REAL, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_Z2D:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_REAL_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_C2C:
                ret = hipfftXtSetCallback(plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            case HIPFFT_Z2Z:
                ret = hipfftXtSetCallback(
                    plan, &load_cb_host, HIPFFT_CB_LD_COMPLEX_DOUBLE, &load_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);

                ret = hipfftXtSetCallback(
                    plan, &store_cb_host, HIPFFT_CB_ST_COMPLEX_DOUBLE, &store_cb_data);
                if(ret != HIPFFT_SUCCESS)
                    return fft_status_from_hipfftparams(ret);
                break;
            default:
                throw std::runtime_error("Invalid execution type");
            }
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

        // if we're doing multi-GPU, we need to use ExecDescriptor
        // methods to execute.
        if(!ifields.empty() || !ofields.empty())
        {
            // rotate between generic ExecDescriptor and specific
            // ExecDescriptorX2Y functions by hashing token (for
            // stability across reruns of test cases)
            //
            // the specific functions are only for the main transform
            // types expressible through the hipfftType enum
            bool generic_ExecDescriptor
                = !hipfft_transform_type || std::hash<std::string>()(token()) % 2;

            if(generic_ExecDescriptor)
            {
                ret = hipfftXtExecDescriptor(plan,
                                             placement == fft_placement_inplace ? xt_output.get()
                                                                                : xt_input.get(),
                                             xt_output.get(),
                                             direction);
            }
            else
            {
                switch(*hipfft_transform_type)
                {
                case HIPFFT_R2C:
                    ret = hipfftXtExecDescriptorR2C(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_C2R:
                    ret = hipfftXtExecDescriptorC2R(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_C2C:
                    ret = hipfftXtExecDescriptorC2C(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get(),
                        direction);
                    break;
                case HIPFFT_D2Z:
                    ret = hipfftXtExecDescriptorD2Z(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_Z2D:
                    ret = hipfftXtExecDescriptorZ2D(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get());
                    break;
                case HIPFFT_Z2Z:
                    ret = hipfftXtExecDescriptorZ2Z(
                        plan,
                        placement == fft_placement_inplace ? xt_output.get() : xt_input.get(),
                        xt_output.get(),
                        direction);
                }
            }
            return fft_status_from_hipfftparams(ret);
        }

        // otherwise, we have two ways to execute in hipFFT -
        // hipfftExecFOO and hipfftXtExec

        // Transforms that aren't supported by the hipfftType enum
        // require using the Xt method, but otherwise we hash the
        // token to decide how to execute this FFT.  we want test
        // cases to rotate between different execution APIs, but we also
        // need the choice of API to be stable across reruns of the
        // same test cases.
        if(!hipfft_transform_type || std::hash<std::string>()(token()) % 2)
        {
            ret = hipfftXtExec(plan, ibuffer, obuffer, direction);
        }
        else
        {
            try
            {
                switch(*hipfft_transform_type)
                {
                case HIPFFT_R2C:
                    ret = hipfftExecR2C(
                        plan,
                        (hipfftReal*)ibuffer,
                        (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                    break;
                case HIPFFT_D2Z:
                    ret = hipfftExecD2Z(plan,
                                        (hipfftDoubleReal*)ibuffer,
                                        (hipfftDoubleComplex*)(placement == fft_placement_inplace
                                                                   ? ibuffer
                                                                   : obuffer));
                    break;
                case HIPFFT_C2R:
                    ret = hipfftExecC2R(
                        plan,
                        (hipfftComplex*)ibuffer,
                        (hipfftReal*)(placement == fft_placement_inplace ? ibuffer : obuffer));
                    break;
                case HIPFFT_Z2D:
                    ret = hipfftExecZ2D(plan,
                                        (hipfftDoubleComplex*)ibuffer,
                                        (hipfftDoubleReal*)(placement == fft_placement_inplace
                                                                ? ibuffer
                                                                : obuffer));
                    break;
                case HIPFFT_C2C:
                    ret = hipfftExecC2C(
                        plan,
                        (hipfftComplex*)ibuffer,
                        (hipfftComplex*)(placement == fft_placement_inplace ? ibuffer : obuffer),
                        direction);
                    break;
                case HIPFFT_Z2Z:
                    ret = hipfftExecZ2Z(plan,
                                        (hipfftDoubleComplex*)ibuffer,
                                        (hipfftDoubleComplex*)(placement == fft_placement_inplace
                                                                   ? ibuffer
                                                                   : obuffer),
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
                std::cerr << "unknown exception in execute(void* ibuffer, void* obuffer)"
                          << std::endl;
            }
        }
        return fft_status_from_hipfftparams(ret);
    }

    bool is_contiguous() const
    {
        // compute contiguous stride, dist and check that the actual
        // strides/dists match
        std::vector<size_t> contiguous_istride
            = compute_stride(ilength(),
                             {},
                             placement == fft_placement_inplace
                                 && transform_type == fft_transform_type_real_forward);
        std::vector<size_t> contiguous_ostride
            = compute_stride(olength(),
                             {},
                             placement == fft_placement_inplace
                                 && transform_type == fft_transform_type_real_inverse);
        if(istride != contiguous_istride || ostride != contiguous_ostride)
            return false;
        return compute_idist() == idist && compute_odist() == odist;
    }

    // call the hipFFT APIs to distribute data to multiple GPUs
    void multi_gpu_prepare(std::vector<gpubuf>& ibuffer,
                           std::vector<void*>&  pibuffer,
                           std::vector<void*>&  pobuffer) override
    {
        if(ifields.empty() && ofields.empty())
            return;

        // input data is on the device - copy it back to the host so
        // hipfftXtMemcpy can deal with it
        hostbuf input_host;
        input_host.alloc(ibuffer.front().size());
        if(hipMemcpy(input_host.data(),
                     ibuffer.front().data(),
                     ibuffer.front().size(),
                     hipMemcpyDeviceToHost)
           != hipSuccess)
            throw std::runtime_error("copy back to host failed");

        // allocate data on the multiple GPUs
        if(placement == fft_placement_inplace)
        {
            hipLibXtDesc* xt_tmp = nullptr;
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_INPLACE) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_output.reset(xt_tmp);
            xt_tmp = nullptr;

            if(hipfftXtMemcpy(plan, xt_output.get(), input_host.data(), HIPFFT_COPY_HOST_TO_DEVICE)
               != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMemcpy failed");

            pibuffer.clear();
            std::copy_n(xt_output->descriptor->data,
                        xt_output->descriptor->nGPUs,
                        std::back_inserter(pibuffer));
            pobuffer.clear();
        }
        else
        {
            hipLibXtDesc* xt_tmp = nullptr;
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_INPLACE) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_input.reset(xt_tmp);
            xt_tmp = nullptr;

            if(hipfftXtMemcpy(plan, xt_input.get(), input_host.data(), HIPFFT_COPY_HOST_TO_DEVICE)
               != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMemcpy failed");
            if(hipfftXtMalloc(plan, &xt_tmp, HIPFFT_XT_FORMAT_OUTPUT) != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftXtMalloc failed");
            xt_output.reset(xt_tmp);
            xt_tmp = nullptr;

            pibuffer.clear();
            std::copy_n(xt_input->descriptor->data,
                        xt_input->descriptor->nGPUs,
                        std::back_inserter(pibuffer));
            pobuffer.clear();
            std::copy_n(xt_output->descriptor->data,
                        xt_output->descriptor->nGPUs,
                        std::back_inserter(pobuffer));
        }
    }

    // call the hipFFT APIs to gather the data back from the multiple GPUs
    virtual void multi_gpu_finalize(std::vector<gpubuf>& obuffer,
                                    std::vector<void*>&  pobuffer) override
    {
        if(ifields.empty() && ofields.empty())
            return;

        // allocate a host buffer for hipFFTXtMemcpy's sake
        hostbuf output_host;
        output_host.alloc(obuffer.front().size());

        if(hipfftXtMemcpy(plan, output_host.data(), xt_output.get(), HIPFFT_COPY_DEVICE_TO_HOST)
           != HIPFFT_SUCCESS)
            throw std::runtime_error("hipfftXtMemcpy failed");

        // copy final result back to device for comparison
        if(hipMemcpy(obuffer.front().data(),
                     output_host.data(),
                     obuffer.front().size(),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("finalizing hipMemcpy failed");

        pobuffer.clear();
        pobuffer.push_back(obuffer.front().data());
    }

private:
    // hipFFT provides multiple ways to create FFT plans:
    // - hipfftPlan1d/2d/3d (combined allocate + init for specific dim)
    // - hipfftPlanMany (combined allocate + init with dim as param)
    // - hipfftCreate + hipfftMakePlan1d/2d/3d (separate alloc + init for specific dim)
    // - hipfftCreate + hipfftMakePlanMany (separate alloc + init with dim as param)
    // - hipfftCreate + hipfftMakePlanMany64 (separate alloc + init with dim as param, 64-bit)
    // - hipfftCreate + hipfftXtMakePlanMany (separate alloc + init with separate i/o/exec types)
    //
    // Rotate through the choices for better test coverage.
    enum PlanCreateAPI
    {
        PLAN_Nd,
        PLAN_MANY,
        CREATE_MAKE_PLAN_Nd,
        CREATE_MAKE_PLAN_MANY,
        CREATE_MAKE_PLAN_MANY64,
        CREATE_XT_MAKE_PLAN_MANY,
    };

    // return true if we need to use hipFFT APIs that separate plan
    // allocation and plan init
    bool need_separate_create_make() const
    {
        // scale factor and multi-GPU need API calls between create +
        // init
        if(scale_factor != 1.0 || !ifields.empty() || !ofields.empty())
            return true;
        return false;
    }

    // Not all plan options work with all creation types.  Return a
    // suitable plan creation type for the current FFT parameters.
    int get_create_type()
    {
        bool contiguous = is_contiguous();
        bool batched    = nbatch > 1;

        std::vector<PlanCreateAPI> allowed_apis;

        // half-precision requires XtMakePlanMany
        if(precision == fft_precision_half)
        {
            allowed_apis.push_back(CREATE_XT_MAKE_PLAN_MANY);
        }
        else
        {
            // separate alloc + init "Many" APIs are always allowed
            allowed_apis.push_back(CREATE_MAKE_PLAN_MANY);
            allowed_apis.push_back(CREATE_MAKE_PLAN_MANY64);
            allowed_apis.push_back(CREATE_XT_MAKE_PLAN_MANY);

            if(!need_separate_create_make())
                allowed_apis.push_back(PLAN_MANY);

            // non-many APIs are only allowed if FFT is contiguous, and
            // only the 1D API allows for batched FFTs.
            if(contiguous && (!batched || dim() == 1))
            {
                if(!need_separate_create_make())
                    allowed_apis.push_back(PLAN_Nd);
                allowed_apis.push_back(CREATE_MAKE_PLAN_Nd);
            }
        }

        // hash the token to decide how to create this FFT.  we want
        // test cases to rotate between different create APIs, but we
        // also need the choice of API to be stable across reruns of
        // the same test cases.
        return allowed_apis[std::hash<std::string>()(token()) % allowed_apis.size()];
    }

    // call hipfftPlan* functions
    hipfftResult_t create_plan_Nd()
    {
        auto ret = HIPFFT_INVALID_PLAN;
        switch(dim())
        {
        case 1:
            ret = hipfftPlan1d(&plan, int_length[0], *hipfft_transform_type, nbatch);
            break;
        case 2:
            ret = hipfftPlan2d(&plan, int_length[0], int_length[1], *hipfft_transform_type);
            break;
        case 3:
            ret = hipfftPlan3d(
                &plan, int_length[0], int_length[1], int_length[2], *hipfft_transform_type);
            break;
        default:
            throw std::runtime_error("invalid dim");
        }
        return ret;
    }
    hipfftResult_t create_plan_many()
    {
        auto ret = hipfftPlanMany(&plan,
                                  dim(),
                                  int_length.data(),
                                  int_inembed.data(),
                                  istride.back(),
                                  idist,
                                  int_onembed.data(),
                                  ostride.back(),
                                  odist,
                                  *hipfft_transform_type,
                                  nbatch);
        return ret;
    }

    // call hipfftCreate + hipfftMake* functions, inserting calls to
    // relevant pre-Make APIs (scale factor, XtSetGPUs)
    hipfftResult_t create_with_pre_make()
    {
        auto ret = hipfftCreate(&plan);
        if(ret != HIPFFT_SUCCESS)
            return ret;
        if(scale_factor != 1.0)
        {
            ret = hipfftExtPlanScaleFactor(plan, scale_factor);
            if(ret != HIPFFT_SUCCESS)
                return ret;
        }
        if(!ifields.empty() || !ofields.empty())
        {
            int deviceCount = 0;
            (void)hipGetDeviceCount(&deviceCount);

            std::vector<int> GPUs(static_cast<size_t>(deviceCount));
            std::iota(GPUs.begin(), GPUs.end(), 0);
            ret = hipfftXtSetGPUs(plan, deviceCount, GPUs.data());

            xt_worksize.resize(GPUs.size());
            workbuffersize_ptr = xt_worksize.data();
        }
        return ret;
    }
    hipfftResult_t create_make_plan_Nd()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;

        switch(dim())
        {
        case 1:
            return hipfftMakePlan1d(
                plan, int_length[0], *hipfft_transform_type, nbatch, workbuffersize_ptr);
        case 2:
            return hipfftMakePlan2d(
                plan, int_length[0], int_length[1], *hipfft_transform_type, workbuffersize_ptr);
        case 3:
            return hipfftMakePlan3d(plan,
                                    int_length[0],
                                    int_length[1],
                                    int_length[2],
                                    *hipfft_transform_type,
                                    workbuffersize_ptr);
        default:
            throw std::runtime_error("invalid dim");
        }
    }
    hipfftResult_t create_make_plan_many()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;
        return hipfftMakePlanMany(plan,
                                  dim(),
                                  int_length.data(),
                                  int_inembed.data(),
                                  istride.back(),
                                  idist,
                                  int_onembed.data(),
                                  ostride.back(),
                                  odist,
                                  *hipfft_transform_type,
                                  nbatch,
                                  workbuffersize_ptr);
    }
    hipfftResult_t create_make_plan_many64()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;
        return hipfftMakePlanMany64(plan,
                                    dim(),
                                    ll_length.data(),
                                    ll_inembed.data(),
                                    istride.back(),
                                    idist,
                                    ll_onembed.data(),
                                    ostride.back(),
                                    odist,
                                    *hipfft_transform_type,
                                    nbatch,
                                    workbuffersize_ptr);
    }

    hipfftResult_t create_xt_make_plan_many()
    {
        auto ret = create_with_pre_make();
        if(ret != HIPFFT_SUCCESS)
            return ret;

        // execution type is always complex, matching the precision
        // of the transform
        // Initializing as double by default
        hipDataType executionType = HIP_C_64F;
        switch(precision)
        {
        case fft_precision_half:
            executionType = HIP_C_16F;
            break;
        case fft_precision_single:
            executionType = HIP_C_32F;
            break;
        case fft_precision_double:
            executionType = HIP_C_64F;
            break;
        }

        return hipfftXtMakePlanMany(plan,
                                    dim(),
                                    ll_length.data(),
                                    ll_inembed.data(),
                                    istride.back(),
                                    idist,
                                    inputType,
                                    ll_onembed.data(),
                                    ostride.back(),
                                    odist,
                                    outputType,
                                    nbatch,
                                    workbuffersize_ptr,
                                    executionType);
    }
};

#endif
