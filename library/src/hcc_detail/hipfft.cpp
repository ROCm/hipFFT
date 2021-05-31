// Copyright (c) 2016 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipfft.h"
#include "hipfftXt.h"
#include "rocfft.h"
#include <sstream>
#include <string>
#include <vector>

#define ROC_FFT_CHECK_ALLOC_FAILED(ret)  \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_ALLOC_FAILED;  \
        }                                \
    }

#define ROC_FFT_CHECK_INVALID_VALUE(ret) \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_INVALID_VALUE; \
        }                                \
    }

#define ROC_FFT_CHECK_EXEC_FAILED(ret)   \
    {                                    \
        if(ret != rocfft_status_success) \
        {                                \
            return HIPFFT_EXEC_FAILED;   \
        }                                \
    }

#define HIP_FFT_CHECK_AND_RETURN(ret) \
    {                                 \
        if(ret != HIPFFT_SUCCESS)     \
        {                             \
            return ret;               \
        }                             \
    }

struct hipfftHandle_t
{
    hipfftType type = HIPFFT_C2C;

    // Due to hipExec** compatibility to cuFFT, we have to reserve all 4 types
    // rocfft handle separately here.
    rocfft_plan           ip_forward          = nullptr;
    rocfft_plan           op_forward          = nullptr;
    rocfft_plan           ip_inverse          = nullptr;
    rocfft_plan           op_inverse          = nullptr;
    rocfft_execution_info info                = nullptr;
    void*                 workBuffer          = nullptr;
    size_t                workBufferSize      = 0;
    bool                  autoAllocate        = true;
    bool                  workBufferNeedsFree = false;

    void** load_callback_ptrs       = nullptr;
    void** load_callback_data       = nullptr;
    size_t load_callback_lds_bytes  = 0;
    void** store_callback_ptrs      = nullptr;
    void** store_callback_data      = nullptr;
    size_t store_callback_lds_bytes = 0;
};

struct hipfft_plan_description_t
{
    rocfft_array_type inArrayType, outArrayType;

    size_t inStrides[3];
    size_t outStrides[3];

    size_t inDist;
    size_t outDist;

    hipfft_plan_description_t()
    {
        inArrayType  = rocfft_array_type_complex_interleaved;
        outArrayType = rocfft_array_type_complex_interleaved;

        inDist  = 0;
        outDist = 0;
    }
};

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan1d(*plan, nx, type, batch, nullptr);
}

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan2d(*plan, nx, ny, type, nullptr);
}

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan3d(*plan, nx, ny, nz, type, nullptr);
}

hipfftResult hipfftPlanMany(hipfftHandle* plan,
                            int           rank,
                            int*          n,
                            int*          inembed,
                            int           istride,
                            int           idist,
                            int*          onembed,
                            int           ostride,
                            int           odist,
                            hipfftType    type,
                            int           batch)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlanMany(
        *plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, nullptr);
}

hipfftResult hipfftMakePlan_internal(hipfftHandle               plan,
                                     size_t                     dim,
                                     size_t*                    lengths,
                                     hipfftType                 type,
                                     size_t                     number_of_transforms,
                                     hipfft_plan_description_t* desc,
                                     size_t*                    workSize,
                                     bool                       re_calc_strides_in_desc)
{
    // magic static to handle rocfft setup/cleanup
    struct rocfft_initializer
    {
        rocfft_initializer()
        {
            rocfft_setup();
        }
        ~rocfft_initializer()
        {
            rocfft_cleanup();
        }
    };
    static rocfft_initializer init;

    rocfft_plan_description ip_forward_desc = nullptr;
    rocfft_plan_description op_forward_desc = nullptr;
    rocfft_plan_description ip_inverse_desc = nullptr;
    rocfft_plan_description op_inverse_desc = nullptr;

    if(desc != nullptr)
    {
        rocfft_plan_description_create(&ip_forward_desc);
        rocfft_plan_description_create(&op_forward_desc);
        rocfft_plan_description_create(&ip_inverse_desc);
        rocfft_plan_description_create(&op_inverse_desc);

        size_t i_strides[3] = {desc->inStrides[0], desc->inStrides[1], desc->inStrides[2]};
        size_t o_strides[3] = {desc->outStrides[0], desc->outStrides[1], desc->outStrides[2]};

        if(re_calc_strides_in_desc)
        {
            if(desc->inArrayType == rocfft_array_type_real) // real-to-complex
            {
                size_t idist = 2 * (1 + lengths[0] / 2);
                size_t odist = 1 + lengths[0] / 2;
                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = idist;
                    idist *= lengths[i];
                    o_strides[i] = odist;
                    odist *= lengths[i];
                }

                desc->inDist  = idist;
                desc->outDist = odist;

                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(ip_forward_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));

                idist = lengths[0];
                odist = 1 + lengths[0] / 2;
                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = idist;
                    idist *= lengths[i];
                    o_strides[i] = odist;
                    odist *= lengths[i];
                }

                desc->inDist  = idist;
                desc->outDist = odist;

                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(op_forward_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
            }
            else if(desc->outArrayType == rocfft_array_type_real) // complex-to-real
            {
                size_t idist = 1 + lengths[0] / 2;
                size_t odist = 2 * (1 + lengths[0] / 2);
                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = idist;
                    idist *= lengths[i];
                    o_strides[i] = odist;
                    odist *= lengths[i];
                }

                desc->inDist  = idist;
                desc->outDist = odist;

                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(ip_inverse_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));

                idist = 1 + lengths[0] / 2;
                odist = lengths[0];
                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = idist;
                    idist *= lengths[i];
                    o_strides[i] = odist;
                    odist *= lengths[i];
                }

                desc->inDist  = idist;
                desc->outDist = odist;

                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(op_inverse_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
            }
            else
            {

                size_t dist = lengths[0];
                for(size_t i = 1; i < dim; i++)
                {
                    dist *= lengths[i];
                }

                desc->inDist  = dist;
                desc->outDist = dist;

                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(ip_forward_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(op_forward_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(ip_inverse_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
                ROC_FFT_CHECK_INVALID_VALUE(
                    rocfft_plan_description_set_data_layout(op_inverse_desc,
                                                            desc->inArrayType,
                                                            desc->outArrayType,
                                                            0,
                                                            0,
                                                            dim,
                                                            i_strides,
                                                            desc->inDist,
                                                            dim,
                                                            o_strides,
                                                            desc->outDist));
            }
        }
        else
        {
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_data_layout(ip_forward_desc,
                                                                                desc->inArrayType,
                                                                                desc->outArrayType,
                                                                                0,
                                                                                0,
                                                                                dim,
                                                                                i_strides,
                                                                                desc->inDist,
                                                                                dim,
                                                                                o_strides,
                                                                                desc->outDist));
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_data_layout(op_forward_desc,
                                                                                desc->inArrayType,
                                                                                desc->outArrayType,
                                                                                0,
                                                                                0,
                                                                                dim,
                                                                                i_strides,
                                                                                desc->inDist,
                                                                                dim,
                                                                                o_strides,
                                                                                desc->outDist));
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_data_layout(ip_inverse_desc,
                                                                                desc->inArrayType,
                                                                                desc->outArrayType,
                                                                                0,
                                                                                0,
                                                                                dim,
                                                                                i_strides,
                                                                                desc->inDist,
                                                                                dim,
                                                                                o_strides,
                                                                                desc->outDist));
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_description_set_data_layout(op_inverse_desc,
                                                                                desc->inArrayType,
                                                                                desc->outArrayType,
                                                                                0,
                                                                                0,
                                                                                dim,
                                                                                i_strides,
                                                                                desc->inDist,
                                                                                dim,
                                                                                o_strides,
                                                                                desc->outDist));
        }
    }

    switch(type)
    {
    case HIPFFT_R2C:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_forward,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_real_forward,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_forward,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_real_forward,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_forward_desc));
        break;
    case HIPFFT_C2R:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_inverse,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_real_inverse,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_inverse_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_inverse,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_real_inverse,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_inverse_desc));
        break;
    case HIPFFT_C2C:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_forward,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_complex_forward,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_forward,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_complex_forward,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_inverse,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_complex_inverse,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_inverse_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_inverse,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_complex_inverse,
                                                       rocfft_precision_single,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_inverse_desc));
        break;

    case HIPFFT_D2Z:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_forward,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_real_forward,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_forward,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_real_forward,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_forward_desc));
        break;
    case HIPFFT_Z2D:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_inverse,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_real_inverse,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_inverse_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_inverse,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_real_inverse,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_inverse_desc));
        break;
    case HIPFFT_Z2Z:
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_forward,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_complex_forward,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_forward,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_complex_forward,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_forward_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->ip_inverse,
                                                       rocfft_placement_inplace,
                                                       rocfft_transform_type_complex_inverse,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       ip_inverse_desc));
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_create(&plan->op_inverse,
                                                       rocfft_placement_notinplace,
                                                       rocfft_transform_type_complex_inverse,
                                                       rocfft_precision_double,
                                                       dim,
                                                       lengths,
                                                       number_of_transforms,
                                                       op_inverse_desc));
        break;
    default:
        return HIPFFT_PARSE_ERROR;
    }
    plan->type = type;

    size_t workBufferSize = 0;
    size_t tmpBufferSize  = 0;

    bool const has_forward = type != HIPFFT_C2R && type != HIPFFT_Z2D;
    if(has_forward)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }

    bool const has_inverse = type != HIPFFT_R2C && type != HIPFFT_D2Z;
    if(has_inverse)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }

    if(workBufferSize > 0)
    {
        if(plan->autoAllocate)
        {
            if(plan->workBuffer && plan->workBufferNeedsFree)
                if(hipFree(plan->workBuffer) != hipSuccess)
                    return HIPFFT_ALLOC_FAILED;
            if(hipMalloc(&plan->workBuffer, workBufferSize) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
            plan->workBufferNeedsFree = true;
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_work_buffer(
                plan->info, plan->workBuffer, workBufferSize));
        }
    }

    if(workSize != nullptr)
        *workSize = workBufferSize;

    plan->workBufferSize = workBufferSize;

    rocfft_plan_description_destroy(ip_forward_desc);
    rocfft_plan_description_destroy(op_forward_desc);
    rocfft_plan_description_destroy(ip_inverse_desc);
    rocfft_plan_description_destroy(op_inverse_desc);

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftCreate(hipfftHandle* plan)
{
    hipfftHandle h = new hipfftHandle_t;
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_create(&h->info));
    *plan = h;
    return HIPFFT_SUCCESS;
}

hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{

    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[1];
    lengths[0]                                      = nx;
    size_t                     number_of_transforms = batch;
    hipfft_plan_description_t* desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 1, lengths, type, number_of_transforms, desc, workSize, false);
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{

    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[2];
    lengths[0]                                      = ny;
    lengths[1]                                      = nx;
    size_t                     number_of_transforms = 1;
    hipfft_plan_description_t* desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 2, lengths, type, number_of_transforms, desc, workSize, false);
}

hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{

    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    size_t lengths[3];
    lengths[0]                                      = nz;
    lengths[1]                                      = ny;
    lengths[2]                                      = nx;
    size_t                     number_of_transforms = 1;
    hipfft_plan_description_t* desc                 = nullptr;

    return hipfftMakePlan_internal(
        plan, 3, lengths, type, number_of_transforms, desc, workSize, false);
}

hipfftResult hipfftMakePlanMany(hipfftHandle plan,
                                int          rank,
                                int*         n,
                                int*         inembed,
                                int          istride,
                                int          idist,
                                int*         onembed,
                                int          ostride,
                                int          odist,
                                hipfftType   type,
                                int          batch,
                                size_t*      workSize)
{
    size_t lengths[3];
    for(size_t i = 0; i < rank; i++)
        lengths[i] = n[rank - 1 - i];

    size_t number_of_transforms = batch;

    // Decide the inArrayType and outArrayType based on the transform type
    rocfft_array_type in_array_type, out_array_type;
    switch(type)
    {
    case HIPFFT_R2C:
    case HIPFFT_D2Z:
        in_array_type  = rocfft_array_type_real;
        out_array_type = rocfft_array_type_hermitian_interleaved;
        break;
    case HIPFFT_C2R:
    case HIPFFT_Z2D:
        in_array_type  = rocfft_array_type_hermitian_interleaved;
        out_array_type = rocfft_array_type_real;
        break;
    case HIPFFT_C2C:
    case HIPFFT_Z2Z:
        in_array_type  = rocfft_array_type_complex_interleaved;
        out_array_type = rocfft_array_type_complex_interleaved;
        break;
    }

    hipfft_plan_description_t desc;

    bool re_calc_strides_in_desc = (inembed == nullptr) || (onembed == nullptr);

    size_t i_strides[3] = {1, 1, 1};
    size_t o_strides[3] = {1, 1, 1};
    for(size_t i = 1; i < rank; i++)
    {
        i_strides[i] = lengths[i - 1] * i_strides[i - 1];
        o_strides[i] = lengths[i - 1] * o_strides[i - 1];
    }

    if(inembed != nullptr)
    {
        i_strides[0] = istride;

        size_t inembed_lengths[3];
        for(size_t i = 0; i < rank; i++)
            inembed_lengths[i] = inembed[rank - 1 - i];

        for(size_t i = 1; i < rank; i++)
            i_strides[i] = inembed_lengths[i - 1] * i_strides[i - 1];
    }

    if(onembed != nullptr)
    {
        o_strides[0] = ostride;

        size_t onembed_lengths[3];
        for(size_t i = 0; i < rank; i++)
            onembed_lengths[i] = onembed[rank - 1 - i];

        for(size_t i = 1; i < rank; i++)
            o_strides[i] = onembed_lengths[i - 1] * o_strides[i - 1];
    }

    desc.inArrayType  = in_array_type;
    desc.outArrayType = out_array_type;

    for(size_t i = 0; i < rank; i++)
        desc.inStrides[i] = i_strides[i];
    desc.inDist = idist;

    for(size_t i = 0; i < rank; i++)
        desc.outStrides[i] = o_strides[i];
    desc.outDist = odist;

    hipfftResult ret = hipfftMakePlan_internal(
        plan, rank, lengths, type, number_of_transforms, &desc, workSize, re_calc_strides_in_desc);

    return ret;
}

hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
                                  int            rank,
                                  long long int* n,
                                  long long int* inembed,
                                  long long int  istride,
                                  long long int  idist,
                                  long long int* onembed,
                                  long long int  ostride,
                                  long long int  odist,
                                  hipfftType     type,
                                  long long int  batch,
                                  size_t*        workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
{
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize1d(plan, nx, type, batch, workSize);
    return ret;
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
{
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize2d(plan, nx, ny, type, workSize);
    return ret;
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSize3d(plan, nx, ny, nz, type, workSize);
    return ret;
}

hipfftResult hipfftEstimateMany(int        rank,
                                int*       n,
                                int*       inembed,
                                int        istride,
                                int        idist,
                                int*       onembed,
                                int        ostride,
                                int        odist,
                                hipfftType type,
                                int        batch,
                                size_t*    workSize)
{
    hipfftHandle plan = nullptr;
    hipfftResult ret  = hipfftGetSizeMany(
        plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
    return ret;
}

hipfftResult hipfftGetSize_internal(hipfftHandle plan, hipfftType type, size_t* workSize)
{

    if(type == HIPFFT_C2C || type == HIPFFT_Z2Z) // TODO
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }
    else if(type == HIPFFT_C2R || type == HIPFFT_Z2D)
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }
    else // R2C or D2Z
    {
        ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_get_work_buffer_size(plan->op_forward, workSize));
    }

    return HIPFFT_SUCCESS;
}

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{

    if(nx < 0 || batch < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan1d(p, nx, type, batch, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    if(nx < 0 || ny < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan2d(p, nx, ny, type, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    if(nx < 0 || ny < 0 || nz < 0)
    {
        return HIPFFT_INVALID_SIZE;
    }

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&p));
    HIP_FFT_CHECK_AND_RETURN(hipfftMakePlan3d(p, nx, ny, nz, type, workSize));
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSizeMany(hipfftHandle plan,
                               int          rank,
                               int*         n,
                               int*         inembed,
                               int          istride,
                               int          idist,
                               int*         onembed,
                               int          ostride,
                               int          odist,
                               hipfftType   type,
                               int          batch,
                               size_t*      workSize)
{

    hipfftHandle p;
    HIP_FFT_CHECK_AND_RETURN(
        hipfftPlanMany(&p, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    *workSize = p->workBufferSize;
    HIP_FFT_CHECK_AND_RETURN(hipfftDestroy(p));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize)
{
    *workSize = plan->workBufferSize;
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
                                 int            rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int  istride,
                                 long long int  idist,
                                 long long int* onembed,
                                 long long int  ostride,
                                 long long int  odist,
                                 hipfftType     type,
                                 long long int  batch,
                                 size_t*        workSize)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    if(plan != nullptr)
        plan->autoAllocate = bool(autoAllocate);
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
{
    if(plan->workBuffer && plan->workBufferNeedsFree)
        hipFree(plan->workBuffer);
    plan->workBufferNeedsFree = false;
    if(workArea)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_execution_info_set_work_buffer(plan->info, workArea, plan->workBufferSize));
    }
    return HIPFFT_SUCCESS;
}

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
{
    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    if(direction == HIPFFT_FORWARD)
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_forward : plan->op_forward, in, out, plan->info));
    }
    else
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_forward : plan->op_forward, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    if(direction == HIPFFT_FORWARD)
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_forward : plan->op_forward, in, out, plan->info));
    }
    else
    {
        ROC_FFT_CHECK_EXEC_FAILED(rocfft_execute(
            idata == odata ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_forward : plan->op_forward, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{

    void* in[1];
    in[0] = (void*)idata;

    void* out[1];
    out[0] = (void*)odata;

    ROC_FFT_CHECK_EXEC_FAILED(
        rocfft_execute(in[0] == out[0] ? plan->ip_inverse : plan->op_inverse, in, out, plan->info));

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_set_stream(plan->info, stream));
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    if(plan != nullptr)
    {
        if(plan->ip_forward != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_forward));
        if(plan->op_forward != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_forward));
        if(plan->ip_inverse != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->ip_inverse));
        if(plan->op_inverse != nullptr)
            ROC_FFT_CHECK_INVALID_VALUE(rocfft_plan_destroy(plan->op_inverse));

        if(plan->workBufferNeedsFree)
            hipFree(plan->workBuffer);

        ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_destroy(plan->info));

        delete plan;
    }

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetVersion(int* version)
{
    char v[256];
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_get_version_string(v, 256));

    // export major.minor.patch only, ignore tweak
    std::ostringstream       result;
    std::vector<std::string> sections;

    std::istringstream iss(v);
    std::string        tmp_str;
    while(std::getline(iss, tmp_str, '.'))
    {
        sections.push_back(tmp_str);
    }

    for(size_t i = 0; i < sections.size() - 1; i++)
    {
        if(sections[i].size() == 1)
            result << "0" << sections[i];
        else
            result << sections[i];
    }

    *version = std::stoi(result.str());
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value)
{
    int full;
    hipfftGetVersion(&full);

    int major = full / 10000;
    int minor = (full - major * 10000) / 100;
    int patch = (full - major * 10000 - minor * 100);

    if(type == HIPFFT_MAJOR_VERSION)
        *value = major;
    else if(type == HIPFFT_MINOR_VERSION)
        *value = minor;
    else if(type == HIPFFT_PATCH_LEVEL)
        *value = patch;
    else
        return HIPFFT_INVALID_TYPE;

    return HIPFFT_SUCCESS;
}

hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                 void**               callbacks,
                                 hipfftXtCallbackType cbtype,
                                 void**               callbackData)
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    // check that the input/output type matches what's being requested
    //
    // NOTE: cufft explicitly does not save shared memory bytes when
    // you set a new callback, so zero out our number when setting
    // pointers
    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
        if(plan->type != HIPFFT_C2R && plan->type != HIPFFT_C2C)
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        if(plan->type != HIPFFT_Z2D && plan->type != HIPFFT_Z2Z)
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL:
        if(plan->type != HIPFFT_R2C)
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        if(plan->type != HIPFFT_D2Z)
            return HIPFFT_INVALID_VALUE;
        plan->load_callback_ptrs      = callbacks;
        plan->load_callback_data      = callbackData;
        plan->load_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX:
        if(plan->type != HIPFFT_R2C && plan->type != HIPFFT_C2C)
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        if(plan->type != HIPFFT_D2Z && plan->type != HIPFFT_Z2Z)
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL:
        if(plan->type != HIPFFT_C2R)
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        if(plan->type != HIPFFT_Z2D)
            return HIPFFT_INVALID_VALUE;
        plan->store_callback_ptrs      = callbacks;
        plan->store_callback_data      = callbackData;
        plan->store_callback_lds_bytes = 0;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

#ifdef ROCFFT_CALLBACKS_ENABLED
    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback_ptrs,
                                                  plan->load_callback_data,
                                                  plan->load_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback_ptrs,
                                                   plan->store_callback_data,
                                                   plan->store_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
#endif
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype)
{
    return hipfftXtSetCallback(plan, nullptr, cbtype, nullptr);
}

hipfftResult
    hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize)
{
    if(!plan)
        return HIPFFT_INVALID_PLAN;

    switch(cbtype)
    {
    case HIPFFT_CB_LD_COMPLEX:
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
    case HIPFFT_CB_LD_REAL:
    case HIPFFT_CB_LD_REAL_DOUBLE:
        plan->load_callback_lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_ST_COMPLEX:
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
    case HIPFFT_CB_ST_REAL:
    case HIPFFT_CB_ST_REAL_DOUBLE:
        plan->store_callback_lds_bytes = sharedSize;
        break;
    case HIPFFT_CB_UNDEFINED:
        return HIPFFT_INVALID_VALUE;
    }

#ifdef ROCFFT_CALLBACKS_ENABLED
    rocfft_status res;
    res = rocfft_execution_info_set_load_callback(plan->info,
                                                  plan->load_callback_ptrs,
                                                  plan->load_callback_data,
                                                  plan->load_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
    res = rocfft_execution_info_set_store_callback(plan->info,
                                                   plan->store_callback_ptrs,
                                                   plan->store_callback_data,
                                                   plan->store_callback_lds_bytes);
    if(res != rocfft_status_success)
        return HIPFFT_INVALID_VALUE;
#endif
    
    return HIPFFT_SUCCESS;
}
