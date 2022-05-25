/******************************************************************************
 * Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights
 * reserved.
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

#ifndef HIPFFTXT_H_
#define HIPFFTXT_H_
#pragma once
#include "hipfft.h"

#ifdef __HIP_PLATFORM_NVIDIA__
#include <cufftXt.h>
#endif

#ifndef _WIN32
#include <cstddef>
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum hipfftXtCallbackType_t
{
    HIPFFT_CB_LD_COMPLEX        = 0x0,
    HIPFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    HIPFFT_CB_LD_REAL           = 0x2,
    HIPFFT_CB_LD_REAL_DOUBLE    = 0x3,
    HIPFFT_CB_ST_COMPLEX        = 0x4,
    HIPFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    HIPFFT_CB_ST_REAL           = 0x6,
    HIPFFT_CB_ST_REAL_DOUBLE    = 0x7,
    HIPFFT_CB_UNDEFINED         = 0x8

} hipfftXtCallbackType;

typedef hipfftComplex (*hipfftCallbackLoadC)(void*  dataIn,
                                             size_t offset,
                                             void*  callerInfo,
                                             void*  sharedPointer);
typedef hipfftDoubleComplex (*hipfftCallbackLoadZ)(void*  dataIn,
                                                   size_t offset,
                                                   void*  callerInfo,
                                                   void*  sharedPointer);
typedef hipfftReal (*hipfftCallbackLoadR)(void*  dataIn,
                                          size_t offset,
                                          void*  callerInfo,
                                          void*  sharedPointer);
typedef hipfftDoubleReal (*hipfftCallbackLoadD)(void*  dataIn,
                                                size_t offset,
                                                void*  callerInfo,
                                                void*  sharedPointer);

typedef void (*hipfftCallbackStoreC)(
    void* dataOut, size_t offset, hipfftComplex element, void* callerInfo, void* sharedPointer);
typedef void (*hipfftCallbackStoreZ)(void*               dataOut,
                                     size_t              offset,
                                     hipfftDoubleComplex element,
                                     void*               callerInfo,
                                     void*               sharedPointer);
typedef void (*hipfftCallbackStoreR)(
    void* dataOut, size_t offset, hipfftReal element, void* callerInfo, void* sharedPointer);
typedef void (*hipfftCallbackStoreD)(
    void* dataOut, size_t offset, hipfftDoubleReal element, void* callerInfo, void* sharedPointer);

/*! @brief Set a callback on a plan
   *
   * @details Set either a load or store callback to run with a plan.
   * The type of callback is specified with the 'cbtype' parameter.
   * An array ofcallback and callback data pointers must be given -
   * one per device executing the plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] callbacks Array of callback function pointers.
   * @param[in] cbtype Type of callback being set.
   * @param[in] callbackData Array of callback function data pointers
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetCallback(hipfftHandle         plan,
                                               void**               callbacks,
                                               hipfftXtCallbackType cbtype,
                                               void**               callbackData);

/*! @brief Remove a callback from a plan
   *
   * @details Remove a previously-set callback from a plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] cbtype Type of callback being removed.
   */
HIPFFT_EXPORT hipfftResult hipfftXtClearCallback(hipfftHandle plan, hipfftXtCallbackType cbtype);

/*! @brief Set shared memory size for callback.
   *
   * @details Set shared memory required for a callback.  The
   * callback of the specified type must have already been set on the
   * plan.
   *
   * @param[in] plan The FFT plan.
   * @param[in] cbtype Type of callback being modified.
   * @param[in] sharedSize Amount of shared memory required, in bytes.
   */
HIPFFT_EXPORT hipfftResult hipfftXtSetCallbackSharedSize(hipfftHandle         plan,
                                                         hipfftXtCallbackType cbtype,
                                                         size_t               sharedSize);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HIPFFT_H_
