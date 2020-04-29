// Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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
    // Due to hipExec** compatibility to cuFFT, we have to reserve all 4 types
    // rocfft handle separately here.
    rocfft_plan           ip_forward;
    rocfft_plan           op_forward;
    rocfft_plan           ip_inverse;
    rocfft_plan           op_inverse;
    rocfft_execution_info info;
    void*                 workBuffer;
    size_t                workBufferSize;
    bool                  autoAllocate;

    hipfftHandle_t()
        : ip_forward(nullptr)
        , op_forward(nullptr)
        , ip_inverse(nullptr)
        , op_inverse(nullptr)
        , info(nullptr)
        , workBuffer(nullptr)
        , workBufferSize(0)
        , autoAllocate(true)
    {
    }
};

// Internal usage
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

/*! \brief Creates a 1D FFT plan configuration for the size and data type. The
 * batch parameter tells how many 1D transforms to perform
 */
hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan1d(*plan, nx, type, batch, nullptr);
}

/*! \brief Creates a 2D FFT plan configuration according to the sizes and data
 * type.
 */
hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
{
    hipfftHandle handle = nullptr;
    HIP_FFT_CHECK_AND_RETURN(hipfftCreate(&handle));
    *plan = handle;

    return hipfftMakePlan2d(*plan, nx, ny, type, nullptr);
}

/*! \brief Creates a 3D FFT plan configuration according to the sizes and data
 * type.
 */
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
    size_t workBufferSize = 0;

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
            if(desc->inArrayType == rocfft_array_type_real) // real-to-complex in-place
            {
                size_t dist = 2 * (1 + lengths[0] / 2);

                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = dist;
                    dist *= lengths[i];
                }

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
            }
            else if(desc->outArrayType == rocfft_array_type_real) // complex-to-real
            {
                size_t dist = 1 + (lengths[0]) / 2;

                for(size_t i = 1; i < dim; i++)
                {
                    i_strides[i] = dist;
                    dist *= lengths[i];
                }
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
            else
            {
                // Set the inStrides to deal with contiguous data
                for(size_t i = 1; i < dim; i++)
                    i_strides[i] = lengths[i - 1] * i_strides[i - 1];

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

    size_t tmpBufferSize = 0;
    if(plan->ip_forward)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->op_forward)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_forward, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->ip_inverse)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->ip_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }
    if(plan->op_inverse)
    {
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_plan_get_work_buffer_size(plan->op_inverse, &tmpBufferSize));
        workBufferSize = std::max(workBufferSize, tmpBufferSize);
    }

    if(workBufferSize > 0)
    {
        if(plan->autoAllocate)
        {
            if(plan->workBuffer)
                if(hipFree(plan->workBuffer) != hipSuccess)
                    return HIPFFT_ALLOC_FAILED;
            if(hipMalloc(&plan->workBuffer, workBufferSize) != hipSuccess)
                return HIPFFT_ALLOC_FAILED;
        }
        ROC_FFT_CHECK_INVALID_VALUE(
            rocfft_execution_info_set_work_buffer(plan->info, plan->workBuffer, workBufferSize));
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

/*===========================================================================*/

hipfftResult hipfftCreate(hipfftHandle* plan)
{
    hipfftHandle h = new hipfftHandle_t;
    ROC_FFT_CHECK_INVALID_VALUE(rocfft_execution_info_create(&h->info));
    *plan = h;
    return HIPFFT_SUCCESS;
}

/*! \brief Assume hipfftCreate has been called. Creates a 1D FFT plan
 * configuration for the size and data type. The batch parameter tells how many
 * 1D transforms to perform
 */
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

/*! \brief Assume hipfftCreate has been called. Creates a 2D FFT plan
 * configuration according to the sizes and data type.
 */
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

/*! \brief Assume hipfftCreate has been called. Creates a 3D FFT plan
 * configuration according to the sizes and data type.
 */
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

/*! \brief

    Creates a FFT plan according to the dimension rank, sizes specified in the
   array n.
    The batch parameter tells hipfft how many transforms to perform. Used in
   complicated usage case like flexbile input & output layout

    \details
    plan 	Pointer to the hipfftHandle object

    rank 	Dimensionality of n.

    n 	    Array of size rank, describing the size of each dimension, n[0]
   being the size of the outermost and n[rank-1] innermost (contiguous)
   dimension of a transform.

    inembed 	Define the number of elements in each dimension the input array.
                Pointer of size rank that indicates the storage dimensions of
   the input data in memory.
                If set to NULL all other advanced data layout parameters are
   ignored.

    istride 	The distance between two successive input elements in the least
   significant (i.e., innermost) dimension

    idist 	    The distance between the first element of two consecutive
   matrices/vetors in a batch of the input data

    onembed 	Define the number of elements in each dimension the output
   array.
                Pointer of size rank that indicates the storage dimensions of
   the output data in memory.
                If set to NULL all other advanced data layout parameters are
   ignored.

    ostride 	The distance between two successive output elements in the
   output array in the least significant (i.e., innermost) dimension

    odist 	    The distance between the first element of two consecutive
   matrices/vectors in a batch of the output data

    batch 	    number of transforms
 */
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

    bool re_calc_strides_in_desc = ((inembed == nullptr) || (onembed == nullptr)) ? true : false;

    size_t i_strides[3] = {1, 1, 1};
    size_t o_strides[3] = {1, 1, 1};

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

    if(idist != 0)
        desc.inDist = idist;

    for(size_t i = 0; i < rank; i++)
        desc.outStrides[i] = o_strides[i];

    if(odist != 0)
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

/*===========================================================================*/

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

/*! \brief gives an accurate estimate of the work area size required for a plan

    Once plan generation has been done, either with the original API or the
   extensible API,
    this call returns the actual size of the work area required to support the
   plan.
    Callers who choose to manage work area allocation within their application
   must use this call after plan generation,
    and after any hipfftSet*() calls subsequent to plan generation, if those
   calls might alter the required work space size.

 */

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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

/*! \brief gives an accurate estimate of the work area size required for a plan
 */

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
    // return hipfftGetSize_internal(plan, type, workArea);

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

/*============================================================================================*/

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    if(plan != nullptr && autoAllocate == 0)
    {
        plan->autoAllocate = false;
    }
    return HIPFFT_SUCCESS;
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
{
    ROC_FFT_CHECK_INVALID_VALUE(
        rocfft_execution_info_set_work_buffer(plan->info, workArea, plan->workBufferSize));
    return HIPFFT_SUCCESS;
}

/*============================================================================================*/

/*! \brief
    executes a single-precision complex-to-complex transform plan in the
   transform direction as specified by direction parameter.
    If idata and odata are the same, this method does an in-place transform,
   otherwise an outofplace transform.
 */
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

/*! \brief
    executes a single-precision real-to-complex, forward, cuFFT transform plan.
 */
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

/*! \brief
    executes a single-precision real-to-complex, inverse, cuFFT transform plan.
 */
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

/*! \brief
    executes a double-precision complex-to-complex transform plan in the
   transform direction as specified by direction parameter.
    If idata and odata are the same, this method does an in-place transform,
   otherwise an outofplace transform.
 */
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

/*! \brief
    executes a double-precision real-to-complex, forward, cuFFT transform plan.
 */
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

/*============================================================================================*/

// Helper functions

/*! \brief
    Associates a HIP stream with a cuFFT plan. All kernel launched with this
   plan execution are associated with this stream
    until the plan is destroyed or the reset to another stream. Returns an error
   in the multiple GPU case as multiple GPU plans perform operations in their
   own streams.
*/
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

        if(plan->autoAllocate)
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
