/******************************************************************************
 * Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights
 *reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/

#include "hipfft.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <hip/hip_runtime.h>

hipfftResult_t cufftResultToHipReult(cufftResult_t cufft_result)
{
    switch(cufft_result)
    {
    case CUFFT_SUCCESS:
        return HIPFFT_SUCCESS;

    case CUFFT_INVALID_PLAN:
        return HIPFFT_INVALID_PLAN;

    case CUFFT_ALLOC_FAILED:
        return HIPFFT_ALLOC_FAILED;

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

hipfftResult hipfftPlan1d(hipfftHandle* plan, int nx, hipfftType type, int batch)
{
    return cufftResultToHipReult(cufftPlan1d(plan, nx, hipfftTypeToCufftType(type), batch));
}

hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type)
{
    return cufftResultToHipReult(cufftPlan2d(plan, nx, ny, hipfftTypeToCufftType(type)));
}

hipfftResult hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type)
{
    return cufftResultToHipReult(cufftPlan3d(plan, nx, ny, nz, hipfftTypeToCufftType(type)));
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
    return cufftResultToHipReult(cufftPlanMany(plan,
                                               rank,
                                               n,
                                               inembed,
                                               istride,
                                               idist,
                                               onembed,
                                               ostride,
                                               odist,
                                               hipfftTypeToCufftType(type),
                                               batch));
}

/*===========================================================================*/

hipfftResult hipfftCreate(hipfftHandle* plan)
{
    return cufftResultToHipReult(cufftCreate(plan));
}

hipfftResult
    hipfftMakePlan1d(hipfftHandle plan, int nx, hipfftType type, int batch, size_t* workSize)
{
    return cufftResultToHipReult(
        cufftMakePlan1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(
        cufftMakePlan2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(
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
    return cufftResultToHipReult(cufftMakePlanMany(plan,
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
    return cufftResultToHipReult(cufftMakePlanMany64(plan,
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
    return cufftResultToHipReult(cufftEstimate1d(nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(cufftEstimate2d(nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(
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
    return cufftResultToHipReult(cufftEstimateMany(rank,
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
    return cufftResultToHipReult(
        cufftGetSize1d(plan, nx, hipfftTypeToCufftType(type), batch, workSize));
}

hipfftResult hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(
        cufftGetSize2d(plan, nx, ny, hipfftTypeToCufftType(type), workSize));
}

hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize)
{
    return cufftResultToHipReult(
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
    return cufftResultToHipReult(cufftGetSizeMany(plan,
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
    return cufftResultToHipReult(cufftGetSizeMany64(plan,
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
    return cufftResultToHipReult(cufftGetSize(plan, workSize));
}

/*===========================================================================*/

hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate)
{
    return cufftResultToHipReult(cufftSetAutoAllocation(plan, autoAllocate));
}

hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea)
{
    return cufftResultToHipReult(cufftSetWorkArea(plan, workArea));
}

/*===========================================================================*/

hipfftResult
    hipfftExecC2C(hipfftHandle plan, hipfftComplex* idata, hipfftComplex* odata, int direction)
{
    return cufftResultToHipReult(cufftExecC2C(plan, idata, odata, direction));
}

hipfftResult hipfftExecR2C(hipfftHandle plan, hipfftReal* idata, hipfftComplex* odata)
{
    return cufftResultToHipReult(cufftExecR2C(plan, idata, odata));
}

hipfftResult hipfftExecC2R(hipfftHandle plan, hipfftComplex* idata, hipfftReal* odata)
{
    return cufftResultToHipReult(cufftExecC2R(plan, idata, odata));
}

hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                           hipfftDoubleComplex* idata,
                           hipfftDoubleComplex* odata,
                           int                  direction)
{
    return cufftResultToHipReult(cufftExecZ2Z(plan, idata, odata, direction));
}

hipfftResult hipfftExecD2Z(hipfftHandle plan, hipfftDoubleReal* idata, hipfftDoubleComplex* odata)
{
    return cufftResultToHipReult(cufftExecD2Z(plan, idata, odata));
}

hipfftResult hipfftExecZ2D(hipfftHandle plan, hipfftDoubleComplex* idata, hipfftDoubleReal* odata)
{
    return cufftResultToHipReult(cufftExecZ2D(plan, idata, odata));
}

/*===========================================================================*/

hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream)
{
    return cufftResultToHipReult(cufftSetStream(plan, stream));
}

hipfftResult hipfftDestroy(hipfftHandle plan)
{
    return cufftResultToHipReult(cufftDestroy(plan));
}

hipfftResult hipfftGetVersion(int* version)
{
    return cufftResultToHipReult(cufftGetVersion(version));
}

hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value)
{
    return cufftResultToHipReult(
        cufftGetProperty(hipfftLibraryPropertyTypeToCufftLibraryPropertyType(type), value));
}
