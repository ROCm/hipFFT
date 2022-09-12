// Copyright (C) 2020 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

hipfftResult_t cufftResultToHipResult(cufftResult_t cufft_result)
{
    switch(cufft_result)
    {
    case CUFFT_SUCCESS:
        return HIPFFT_SUCCESS;

    case CUFFT_INVALID_PLAN:
        return HIPFFT_INVALID_PLAN;

    case CUFFT_ALLOC_FAILED:
        return HIPFFT_ALLOC_FAILED;

    case CUFFT_INVALID_TYPE:
        return HIPFFT_INVALID_TYPE;

    case CUFFT_INVALID_VALUE:
        return HIPFFT_INVALID_VALUE;

    case CUFFT_INTERNAL_ERROR:
        return HIPFFT_INTERNAL_ERROR;

    case CUFFT_EXEC_FAILED:
        return HIPFFT_EXEC_FAILED;

    case CUFFT_SETUP_FAILED:
        return HIPFFT_SETUP_FAILED;

    case CUFFT_INVALID_SIZE:
        return HIPFFT_INVALID_SIZE;

    case CUFFT_UNALIGNED_DATA:
        return HIPFFT_UNALIGNED_DATA;

    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return HIPFFT_INCOMPLETE_PARAMETER_LIST;

    case CUFFT_INVALID_DEVICE:
        return HIPFFT_INVALID_DEVICE;

    case CUFFT_PARSE_ERROR:
        return HIPFFT_PARSE_ERROR;

    case CUFFT_NO_WORKSPACE:
        return HIPFFT_NO_WORKSPACE;

    case CUFFT_NOT_IMPLEMENTED:
        return HIPFFT_NOT_IMPLEMENTED;

    case CUFFT_NOT_SUPPORTED:
        return HIPFFT_NOT_SUPPORTED;

    default:
        throw "Non existent result";
    }
}

cufftType_t hipfftTypeToCufftType(hipfftType_t hipfft_type)
{
    switch(hipfft_type)
    {
    case HIPFFT_R2C:
        return CUFFT_R2C;

    case HIPFFT_C2R:
        return CUFFT_C2R;

    case HIPFFT_C2C:
        return CUFFT_C2C;

    case HIPFFT_D2Z:
        return CUFFT_D2Z;

    case HIPFFT_Z2D:
        return CUFFT_Z2D;

    case HIPFFT_Z2Z:
        return CUFFT_Z2Z;
    default:
        throw "Non existent hipFFT type.";
    }
}

// cudaDataType_t hipDataTypeToCudaDataType(hipDataType hip_data_type)
// {
//     switch(hipfft_type)
//     {
//     case HIP_R_16F:
//         return CUDA_R_16F;

//     case HIP_R_32F:
//         return CUDA_R_32F;

//     case HIP_R_64F:
//         return CUDA_R_64F;

//     case HIP_C_16F:
//         return CUDA_C_16F;

//     case HIP_C_32F:
//         return CUDA_C_32F;

//     case HIP_C_64F:
//         return CUDA_C_64F;

//     default:
//         throw "Not supported hip data type.";
//     }
// }

libraryPropertyType hipfftLibraryPropertyTypeToCufftLibraryPropertyType(
    hipfftLibraryPropertyType_t hipfft_lib_prop_type)
{
    switch(hipfft_lib_prop_type)
    {
    case HIPFFT_MAJOR_VERSION:
        return MAJOR_VERSION;

    case HIPFFT_MINOR_VERSION:
        return MINOR_VERSION;

    case HIPFFT_PATCH_LEVEL:
        return PATCH_LEVEL;

    default:
        throw "Non existent hipFFT library property type.";
    }
}

cufftXtCallbackType_t hipfftCallbackTypeToCufftCallbackType(hipfftXtCallbackType_t type)
{
    switch(type)
    {
    case HIPFFT_CB_LD_COMPLEX:
        return CUFFT_CB_LD_COMPLEX;
    case HIPFFT_CB_LD_COMPLEX_DOUBLE:
        return CUFFT_CB_LD_COMPLEX_DOUBLE;
    case HIPFFT_CB_LD_REAL:
        return CUFFT_CB_LD_REAL;
    case HIPFFT_CB_LD_REAL_DOUBLE:
        return CUFFT_CB_LD_REAL_DOUBLE;
    case HIPFFT_CB_ST_COMPLEX:
        return CUFFT_CB_ST_COMPLEX;
    case HIPFFT_CB_ST_COMPLEX_DOUBLE:
        return CUFFT_CB_ST_COMPLEX_DOUBLE;
    case HIPFFT_CB_ST_REAL:
        return CUFFT_CB_ST_REAL;
    case HIPFFT_CB_ST_REAL_DOUBLE:
        return CUFFT_CB_ST_REAL_DOUBLE;
    case HIPFFT_CB_UNDEFINED:
        return CUFFT_CB_UNDEFINED;
    default:
        throw "Non existent hipFFT XT callback type.";
    }
}

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
{
    return cufftResultToHipResult(cufftPlan1d(plan, nx, hipfftTypeToCufftType(type), batch));
}

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
{
    return cufftResultToHipResult(cufftPlan2d(plan, nx, ny, hipfftTypeToCufftType(type)));
}

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
{
    auto cufftret = CUFFT_SUCCESS;
    try
    {
        cufftret = cufftPlan3d(plan, nx, ny, nz, hipfftTypeToCufftType(type));
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "unknown exception in cufftPlan3d" << std::endl;
    }
    return cufftResultToHipResult(cufftret);
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
    if((inembed == nullptr) != (onembed == nullptr))
    {
        return HIPFFT_INVALID_VALUE;
    }
    else
    {
        auto cufftret = CUFFT_SUCCESS;
        try
        {
            cufftret = cufftPlanMany(plan,
                                     rank,
                                     n,
                                     inembed,
                                     istride,
                                     idist,
                                     onembed,
                                     ostride,
                                     odist,
                                     hipfftTypeToCufftType(type),
                                     batch);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
        catch(...)
        {
            std::cerr << "unknown exception in cufftPlanMany" << std::endl;
        }
        return cufftResultToHipResult(cufftret);
    }
}

/*===========================================================================*/

hipfftResult hipfftCreate(hipfftHandle* plan)
{
    return cufftResultToHipResult(cufftCreate(plan));
}

hipfftResult hipfftExtPlanScaleFactor(hipfftHandle plan, double scalefactor)
{
    return HIPFFT_NOT_IMPLEMENTED;
}

hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftMakePlan1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftMakePlan2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftMakePlan3d(plan, nx, ny, nz, hipfftTypeToCufftType(type), workSize));
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
    if((inembed == nullptr) != (onembed == nullptr))
        return HIPFFT_INVALID_VALUE;

    auto cufftret = CUFFT_SUCCESS;
    try
    {
        cufftret = cufftMakePlanMany(plan,
                                     rank,
                                     n,
                                     inembed,
                                     istride,
                                     idist,
                                     onembed,
                                     ostride,
                                     odist,
                                     hipfftTypeToCufftType(type),
                                     batch,
                                     workSize);
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "unknown exception in cufftMakePlanMany" << std::endl;
    }
    return cufftResultToHipResult(cufftret);
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
    return cufftResultToHipResult(cufftMakePlanMany64(plan,
                                                      rank,
                                                      n,
                                                      inembed,
                                                      istride,
                                                      idist,
                                                      onembed,
                                                      ostride,
                                                      odist,
                                                      hipfftTypeToCufftType(type),
                                                      batch,
                                                      workSize));
}

/*===========================================================================*/

hipfftResult hipfftEstimate1d(int nx, hipfftType type, int batch, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftEstimate1d(nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(cufftEstimate2d(nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftEstimate3d(nx, ny, nz, hipfftTypeToCufftType(type), workSize));
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
    return cufftResultToHipResult(cufftEstimateMany(rank,
                                                    n,
                                                    inembed,
                                                    istride,
                                                    idist,
                                                    onembed,
                                                    ostride,
                                                    odist,
                                                    hipfftTypeToCufftType(type),
                                                    batch,
                                                    workSize));
}

/*===========================================================================*/

hipfftResult
    hipfftGetSize1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftGetSize1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftGetSize2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipResult(
        cufftGetSize3d(plan, nx, ny, nz, hipfftTypeToCufftType(type), workSize));
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
    return cufftResultToHipResult(cufftGetSizeMany(plan,
                                                   rank,
                                                   n,
                                                   inembed,
                                                   istride,
                                                   idist,
                                                   onembed,
                                                   ostride,
                                                   odist,
                                                   hipfftTypeToCufftType(type),
                                                   batch,
                                                   workSize));
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
    return cufftResultToHipResult(cufftGetSizeMany64(plan,
                                                     rank,
                                                     n,
                                                     inembed,
                                                     istride,
                                                     idist,
                                                     onembed,
                                                     ostride,
                                                     odist,
                                                     hipfftTypeToCufftType(type),
                                                     batch,
                                                     workSize));
}

hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize)
{
    return cufftResultToHipResult(cufftGetSize(plan, workSize));
}

/*===========================================================================*/

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    return cufftResultToHipResult(cufftSetAutoAllocation(plan, autoAllocate));
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
{
    return cufftResultToHipResult(cufftSetWorkArea(plan, workArea));
}

/*===========================================================================*/

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
{
    return cufftResultToHipResult(cufftExecC2C(plan, idata, odata, direction));
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    return cufftResultToHipResult(cufftExecR2C(plan, idata, odata));
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    return cufftResultToHipResult(cufftExecC2R(plan, idata, odata));
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
{
    return cufftResultToHipResult(cufftExecZ2Z(plan, idata, odata, direction));
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{
    return cufftResultToHipResult(cufftExecD2Z(plan, idata, odata));
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{
    return cufftResultToHipResult(cufftExecZ2D(plan, idata, odata));
}

/*===========================================================================*/

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    return cufftResultToHipResult(cufftSetStream(plan, stream));
}

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    return cufftResultToHipResult(cufftDestroy(plan));
}

hipfftResult hipfftGetVersion(int* version)
{
    return cufftResultToHipResult(cufftGetVersion(version));
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value)
{
    return cufftResultToHipResult(
        cufftGetProperty(hipfftLibraryPropertyTypeToCufftLibraryPropertyType(type), value));
}

hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                 void**               callbacks,
                                 hipfftXtCallbackType cbtype,
                                 void**               callbackData)
{
    return cufftResultToHipResult(cufftXtSetCallback(
        plan, callbacks, hipfftCallbackTypeToCufftCallbackType(cbtype), callbackData));
}

hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype)
{
    return cufftResultToHipResult(
        cufftXtClearCallback(plan, hipfftCallbackTypeToCufftCallbackType(cbtype)));
}

hipfftResult
    hipfftXtSetCallbackSharedSize(hipfftHandle plan, hipfftXtCallbackType cbtype, size_t sharedSize)
{
    return cufftResultToHipResult(cufftXtSetCallbackSharedSize(
        plan, hipfftCallbackTypeToCufftCallbackType(cbtype), sharedSize));
}
