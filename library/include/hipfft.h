/******************************************************************************
 * Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights
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

#ifndef HIPFFT_H_
#define HIPFFT_H_
#pragma once
#include "hipfft-export.h"
#include "hipfft-version.h"
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <hip/library_types.h>

#ifndef _WIN32
#include <cstddef>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipfftResult_t
{
    HIPFFT_SUCCESS                   = 0, //  The hipFFT operation was successful
    HIPFFT_INVALID_PLAN              = 1, //  hipFFT was passed an invalid plan handle
    HIPFFT_ALLOC_FAILED              = 2, //  hipFFT failed to allocate GPU or CPU memory
    HIPFFT_INVALID_TYPE              = 3, //  No longer used
    HIPFFT_INVALID_VALUE             = 4, //  User specified an invalid pointer or parameter
    HIPFFT_INTERNAL_ERROR            = 5, //  Driver or internal hipFFT library error
    HIPFFT_EXEC_FAILED               = 6, //  Failed to execute an FFT on the GPU
    HIPFFT_SETUP_FAILED              = 7, //  The hipFFT library failed to initialize
    HIPFFT_INVALID_SIZE              = 8, //  User specified an invalid transform size
    HIPFFT_UNALIGNED_DATA            = 9, //  No longer used
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
    HIPFFT_INVALID_DEVICE  = 11, //  Execution of a plan was on different GPU than plan creation
    HIPFFT_PARSE_ERROR     = 12, //  Internal plan database error
    HIPFFT_NO_WORKSPACE    = 13, //  No workspace has been provided prior to plan execution
    HIPFFT_NOT_IMPLEMENTED = 14, // Function does not implement functionality for parameters given.
    HIPFFT_NOT_SUPPORTED   = 16 // Operation is not supported for parameters given.
} hipfftResult;

typedef enum hipfftType_t
{
    HIPFFT_R2C = 0x2a, // Real to complex (interleaved)
    HIPFFT_C2R = 0x2c, // Complex (interleaved) to real
    HIPFFT_C2C = 0x29, // Complex to complex (interleaved)
    HIPFFT_D2Z = 0x6a, // Double to double-complex (interleaved)
    HIPFFT_Z2D = 0x6c, // Double-complex (interleaved) to double
    HIPFFT_Z2Z = 0x69 // Double-complex to double-complex (interleaved)
} hipfftType;

typedef enum hipfftLibraryPropertyType_t
{
    HIPFFT_MAJOR_VERSION,
    HIPFFT_MINOR_VERSION,
    HIPFFT_PATCH_LEVEL
} hipfftLibraryPropertyType;

#define HIPFFT_FORWARD -1
#define HIPFFT_BACKWARD 1

#ifdef __HIP_PLATFORM_NVCC__
typedef int hipfftHandle;
#else
typedef struct hipfftHandle_t* hipfftHandle;
#endif

typedef hipComplex       hipfftComplex;
typedef hipDoubleComplex hipfftDoubleComplex;
typedef float            hipfftReal;
typedef double           hipfftDoubleReal;

HIPFFT_EXPORT hipfftResult hipfftPlan1d(hipfftHandle* plan,
                                        int           nx,
                                        hipfftType    type,
                                        int           batch /* deprecated - use hipfftPlanMany */);

HIPFFT_EXPORT hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type);

HIPFFT_EXPORT hipfftResult
    hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type);

HIPFFT_EXPORT hipfftResult hipfftPlanMany(hipfftHandle* plan,
                                          int           rank,
                                          int*          n,
                                          int*          inembed,
                                          int           istride,
                                          int           idist,
                                          int*          onembed,
                                          int           ostride,
                                          int           odist,
                                          hipfftType    type,
                                          int           batch);

HIPFFT_EXPORT hipfftResult hipfftCreate(hipfftHandle* plan);

HIPFFT_EXPORT hipfftResult hipfftMakePlan1d(hipfftHandle plan,
                                            int          nx,
                                            hipfftType   type,
                                            int     batch, /* deprecated - use hipfftPlanMany */
                                            size_t* workSize);

HIPFFT_EXPORT hipfftResult
    hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult
    hipfftMakePlan3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftMakePlanMany(hipfftHandle plan,
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
                                              size_t*      workSize);

HIPFFT_EXPORT hipfftResult hipfftMakePlanMany64(hipfftHandle   plan,
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
                                                size_t*        workSize);
HIPFFT_EXPORT hipfftResult hipfftEstimate1d(int        nx,
                                            hipfftType type,
                                            int        batch, /* deprecated - use hipfftPlanMany */
                                            size_t*    workSize);

HIPFFT_EXPORT hipfftResult hipfftEstimate2d(int nx, int ny, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult
    hipfftEstimate3d(int nx, int ny, int nz, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftEstimateMany(int        rank,
                                              int*       n,
                                              int*       inembed,
                                              int        istride,
                                              int        idist,
                                              int*       onembed,
                                              int        ostride,
                                              int        odist,
                                              hipfftType type,
                                              int        batch,
                                              size_t*    workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSize1d(hipfftHandle plan,
                                           int          nx,
                                           hipfftType   type,
                                           int     batch, /* deprecated - use hipfftGetSizeMany */
                                           size_t* workSize);

HIPFFT_EXPORT hipfftResult
    hipfftGetSize2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult
    hipfftGetSize3d(hipfftHandle plan, int nx, int ny, int nz, hipfftType type, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSizeMany(hipfftHandle plan,
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
                                             size_t*      workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSizeMany64(hipfftHandle   plan,
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
                                               size_t*        workSize);

HIPFFT_EXPORT hipfftResult hipfftGetSize(hipfftHandle plan, size_t* workSize);

HIPFFT_EXPORT hipfftResult hipfftSetAutoAllocation(hipfftHandle plan, int autoAllocate);

HIPFFT_EXPORT hipfftResult hipfftSetWorkArea(hipfftHandle plan, void* workArea);

HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata,
                                         int            direction);

HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle   plan,
                                         hipfftReal*    idata,
                                         hipfftComplex* odata);

HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftReal*    odata);

HIPFFT_EXPORT hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleComplex* odata,
                                         int                  direction);

HIPFFT_EXPORT hipfftResult hipfftExecD2Z(hipfftHandle         plan,
                                         hipfftDoubleReal*    idata,
                                         hipfftDoubleComplex* odata);

HIPFFT_EXPORT hipfftResult hipfftExecZ2D(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleReal*    odata);

// utility functions
HIPFFT_EXPORT hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);

/*
HIPFFT_EXPORT hipfftResult hipfftSetCompatibilityMode(hipfftHandle plan,
                                               hipfftCompatibility mode);
*/

HIPFFT_EXPORT hipfftResult hipfftDestroy(hipfftHandle plan);

HIPFFT_EXPORT hipfftResult hipfftGetVersion(int* version);

HIPFFT_EXPORT hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HIPFFT_H_
