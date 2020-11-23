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

/*! @brief Result/status/error codes */
typedef enum hipfftResult_t
{
    //! hipFFT operation was successful
    HIPFFT_SUCCESS = 0,
    //! hipFFT was passed an invalid plan handle
    HIPFFT_INVALID_PLAN = 1,
    //! hipFFT failed to allocate GPU or CPU memory
    HIPFFT_ALLOC_FAILED = 2,
    //! No longer used
    HIPFFT_INVALID_TYPE = 3,
    //! User specified an invalid pointer or parameter
    HIPFFT_INVALID_VALUE = 4,
    //! Driver or internal hipFFT library error
    HIPFFT_INTERNAL_ERROR = 5,
    //! Failed to execute an FFT on the GPU
    HIPFFT_EXEC_FAILED = 6,
    //! hipFFT failed to initialize
    HIPFFT_SETUP_FAILED = 7,
    //! User specified an invalid transform size
    HIPFFT_INVALID_SIZE = 8,
    //! No longer used
    HIPFFT_UNALIGNED_DATA = 9,
    //! Missing parameters in call
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    //! Execution of a plan was on different GPU than plan creation
    HIPFFT_INVALID_DEVICE = 11,
    //! Internal plan database error
    HIPFFT_PARSE_ERROR = 12,
    //! No workspace has been provided prior to plan execution
    HIPFFT_NO_WORKSPACE = 13,
    //! Function does not implement functionality for parameters given.
    HIPFFT_NOT_IMPLEMENTED = 14,
    //! Operation is not supported for parameters given.
    HIPFFT_NOT_SUPPORTED = 16
} hipfftResult;

/*! @brief Transform type
 *  @details This type is used to declare the Fourier transform type that will be executed.
 *  */
typedef enum hipfftType_t
{
    //! Real to complex (interleaved)
    HIPFFT_R2C = 0x2a,
    //! Complex (interleaved) to real
    HIPFFT_C2R = 0x2c,
    //! Complex to complex (interleaved)
    HIPFFT_C2C = 0x29,
    //! Double to double-complex (interleaved)
    HIPFFT_D2Z = 0x6a,
    //! Double-complex (interleaved) to double
    HIPFFT_Z2D = 0x6c,
    //! Double-complex to double-complex (interleaved)
    HIPFFT_Z2Z = 0x69
} hipfftType;

typedef enum hipfftLibraryPropertyType_t
{
    HIPFFT_MAJOR_VERSION,
    HIPFFT_MINOR_VERSION,
    HIPFFT_PATCH_LEVEL
} hipfftLibraryPropertyType;

/*! @brief Perform a forward FFT.
 * */
#define HIPFFT_FORWARD -1
/*! @brief Perform a backward/inverse FFT.
 * */
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

/*! @brief Create a new one-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new one-dimensional FFT plan.
 *
 *  @param plan Pointer to the FFT plan.
 *  @param nx FFT length.
 *  @param type FFT type.
 *  @param batch (deprecated) Number of batched transforms to compute.
 *  */
HIPFFT_EXPORT hipfftResult hipfftPlan1d(hipfftHandle* plan,
                                        int           nx,
                                        hipfftType    type,
                                        int           batch /* deprecated - use hipfftPlanMany */);

/*! @brief Create a new two-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new two-dimensional FFT plan.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param plan Pointer to the FFT plan.
 *  @param nx Number of elements in the x-direction (slow index).
 *  @param ny Number of elements in the y-direction (fast index).
 *  @param type FFT type.
 *  */
HIPFFT_EXPORT hipfftResult hipfftPlan2d(hipfftHandle* plan, int nx, int ny, hipfftType type);

/*! @brief Create a new three-dimensional FFT plan.
 *
 *  @details Allocate and initialize a new three-dimensional FFT plan.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param plan Pointer to the FFT plan.
 *  @param nx Number of elements in the x-direction (slowest index).
 *  @param ny Number of elements in the y-direction.
 *  @param nz Number of elements in the z-direction (fastest index).
 *  @param type FFT type.
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftPlan3d(hipfftHandle* plan, int nx, int ny, int nz, hipfftType type);

/*! @brief Create a new batched rank-dimensional FFT plan.
 *
 * @details Allocate and initialize a new batched rank-dimensional
 *  FFT.  The batch parameter tells hipFFT how many transforms to
 *  perform.  Used in complicated usage case like flexbile input and
 *  output layout.
 *
 *  @param plan Pointer to the FFT plan.
 *  @param rank Dimension of FFT transform (1, 2, or 3).
 *  @param n Number of elements in the x/y/z directions.
 *  @param inembed
 *  @param istride
 *  @param idist Distance between input batches.
 *  @param onembed
 *  @param ostride
 *  @param odist Distance between output batches.
 *  @param type FFT type.
 *  @param batch Number of batched transforms to perform.
 *  */
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
/*! @brief Allocate a new plan.
 *  */
HIPFFT_EXPORT hipfftResult hipfftCreate(hipfftHandle* plan);

/*! @brief Initialize a new one-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already.
 *
 *  @param plan Handle of the FFT plan.
 *  @param nx FFT length.
 *  @param type FFT type.
 *  @param batch (deprecated) Number of batched transforms to compute.
 *  */
HIPFFT_EXPORT hipfftResult hipfftMakePlan1d(hipfftHandle plan,
                                            int          nx,
                                            hipfftType   type,
                                            int     batch, /* deprecated - use hipfftPlanMany */
                                            size_t* workSize);

/*! @brief Initialize a new two-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already.
 *  Two-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in y-direction (j index) vary the
 *  fastest.
 *
 *  @param plan Handle of the FFT plan.
 *  @param nx Number of elements in the x-direction (slow index).
 *  @param ny Number of elements in the y-direction (fast index).
 *  @param type FFT type.
 *  @param workSize Pointer to XXX
 *  */
HIPFFT_EXPORT hipfftResult
    hipfftMakePlan2d(hipfftHandle plan, int nx, int ny, hipfftType type, size_t* workSize);

/*! @brief Initialize a new two-dimensional FFT plan.
 *
 *  @details Assumes that the plan has been created already.
 *  Three-dimensional data should be stored in C ordering (row-major
 *  format), so that indexes in z-direction (k index) vary the
 *  fastest.
 *
 *  @param plan Handle of the FFT plan.
 *  @param nx Number of elements in the x-direction (slowest index).
 *  @param ny Number of elements in the y-direction.
 *  @param nz Number of elements in the z-direction (fastest index).
 *  @param type FFT type.
 *  @param workSize Pointer to XXX
 *  */
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

/*! @brief Execute a (float) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 *  @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * */
HIPFFT_EXPORT hipfftResult hipfftExecC2C(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftComplex* odata,
                                         int            direction);

/*! @brief Execute a (float) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecR2C(hipfftHandle   plan,
                                         hipfftReal*    idata,
                                         hipfftComplex* odata);

/*! @brief Execute a (float) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecC2R(hipfftHandle   plan,
                                         hipfftComplex* idata,
                                         hipfftReal*    odata);

/*! @brief Execute a (double) complex-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 *  @param direction Either `HIPFFT_FORWARD` or `HIPFFT_BACKWARD`.
 * */
HIPFFT_EXPORT hipfftResult hipfftExecZ2Z(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleComplex* odata,
                                         int                  direction);

/*! @brief Execute a (double) real-to-complex FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecD2Z(hipfftHandle         plan,
                                         hipfftDoubleReal*    idata,
                                         hipfftDoubleComplex* odata);

/*! @brief Execute a (double) complex-to-real FFT.
 *
 *  @details If the input and output buffers are equal, an in-place
 *  transform is performed.
 *
 *  @param plan The FFT plan.
 *  @param idata Input data (on device).
 *  @param odata Ouput data (on device).
 * */
HIPFFT_EXPORT hipfftResult hipfftExecZ2D(hipfftHandle         plan,
                                         hipfftDoubleComplex* idata,
                                         hipfftDoubleReal*    odata);

/*! @brief Set HIP stream to execute plan on.
 *
 * @details Associates a HIP stream with a hipFFT plan.  All kernels
 * launched by this plan are associated with the provided stream.
 *
 * @param plan The FFT plan.
 * @param stream The HIP stream.
 * */
HIPFFT_EXPORT hipfftResult hipfftSetStream(hipfftHandle plan, hipStream_t stream);

/*
HIPFFT_EXPORT hipfftResult hipfftSetCompatibilityMode(hipfftHandle plan,
                                               hipfftCompatibility mode);
*/

/*! @brief Destroy and deallocate an existing plan.
 *  */
HIPFFT_EXPORT hipfftResult hipfftDestroy(hipfftHandle plan);

/*! @brief Destroy and deallocate an existing plan.
 *  */
HIPFFT_EXPORT hipfftResult hipfftGetVersion(int* version);

HIPFFT_EXPORT hipfftResult hipfftGetProperty(hipfftLibraryPropertyType type, int* value);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HIPFFT_H_
