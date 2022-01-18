// Copyright (c) 2018 - present Advanced Micro Devices, Inc. All rights
// reserved.
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
#include <fftw3.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <vector>

// Function to return maximum error for float and double types.
template <typename Tfloat>
inline double type_epsilon();
template <>
inline double type_epsilon<float>()
{
    return 1e-6;
}
template <>
inline double type_epsilon<double>()
{
    return 1e-7;
}

TEST(hipfftTest, Create1dPlan)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t length = 1024;
    EXPECT_TRUE(hipfftPlan1d(&plan, length, HIPFFT_C2C, 1) == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CreatePlanMany)
{
    int const  rank         = 3;
    int        n[3]         = {64, 128, 23};
    int*       inembed      = n;
    int*       inembed_null = nullptr;
    int const  istride      = 1;
    int const  idist        = 0;
    int*       onembed      = n;
    int*       onembed_null = nullptr;
    int const  ostride      = 1;
    int const  odist        = 0;
    hipfftType type         = HIPFFT_C2C;
    int const  batch        = 1000;
    size_t     workSize;

    hipfftHandle plan_valid_1, plan_valid_2, plan_invalid_1, plan_invalid_2;

    EXPECT_TRUE(hipfftCreate(&plan_valid_1) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftCreate(&plan_valid_2) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftCreate(&plan_invalid_1) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftCreate(&plan_invalid_2) == HIPFFT_SUCCESS);

    // Tests plan creation with null and not null
    // combinations of inembed and onembed.
    //
    // Valid combinations:
    //                      inembed == null && onembed == null
    //                      or
    //                      inembed != null && onembed != null
    //
    // otherwise HIPFFT_INVALID_VALUE should be
    // returned to maintain compatibility with cuFFT

    EXPECT_TRUE(hipfftMakePlanMany(plan_valid_1,
                                   rank,
                                   (int*)n,
                                   inembed_null,
                                   istride,
                                   idist,
                                   onembed_null,
                                   ostride,
                                   odist,
                                   type,
                                   batch,
                                   &workSize)
                == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftMakePlanMany(plan_valid_2,
                                   rank,
                                   (int*)n,
                                   inembed,
                                   istride,
                                   idist,
                                   onembed,
                                   ostride,
                                   odist,
                                   type,
                                   batch,
                                   &workSize)
                == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftMakePlanMany(plan_invalid_1,
                                   rank,
                                   (int*)n,
                                   inembed,
                                   istride,
                                   idist,
                                   onembed_null,
                                   ostride,
                                   odist,
                                   type,
                                   batch,
                                   &workSize)
                == HIPFFT_INVALID_VALUE);

    EXPECT_TRUE(hipfftMakePlanMany(plan_invalid_2,
                                   rank,
                                   (int*)n,
                                   inembed_null,
                                   istride,
                                   idist,
                                   onembed,
                                   ostride,
                                   odist,
                                   type,
                                   batch,
                                   &workSize)
                == HIPFFT_INVALID_VALUE);

    EXPECT_TRUE(hipfftSetAutoAllocation(plan_valid_1, 0) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftSetAutoAllocation(plan_valid_2, 0) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftSetAutoAllocation(plan_invalid_1, 0) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftSetAutoAllocation(plan_invalid_2, 0) == HIPFFT_SUCCESS);

    EXPECT_TRUE(hipfftDestroy(plan_valid_1) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftDestroy(plan_valid_2) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftDestroy(plan_invalid_1) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftDestroy(plan_invalid_2) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2C)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 1024;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_C2C, 1, &workSize) == HIPFFT_SUCCESS);
#ifndef _CUFFT_BACKEND
    // No extra work buffer for C2C
    EXPECT_TRUE(0 == workSize);
#endif
    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeR2C)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_R2C, 1, &workSize) == HIPFFT_SUCCESS);

#ifndef _CUFFT_BACKEND
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(float));
    }
#endif
    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeC2R)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_C2R, 1, &workSize) == HIPFFT_SUCCESS);

#ifndef _CUFFT_BACKEND
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(float));
    }
#endif
    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeD2Z)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_D2Z, batch, &workSize) == HIPFFT_SUCCESS);

#ifndef _CUFFT_BACKEND
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(double));
    }
#endif

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckBufferSizeZ2D)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_Z2D, batch, &workSize) == HIPFFT_SUCCESS);

#ifndef _CUFFT_BACKEND
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(n % 2 == 0)
    {
        EXPECT_TRUE(workSize == 0);
    }
    else
    {
        EXPECT_TRUE(workSize == 2 * n * sizeof(double));
    }
#endif

    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, CheckNullWorkBuffer)
{
    hipfftHandle plan;
    EXPECT_TRUE(hipfftCreate(&plan) == HIPFFT_SUCCESS);
    size_t n        = 2048;
    size_t batch    = 1000;
    size_t workSize = 0;

    EXPECT_TRUE(hipfftMakePlan1d(plan, n, HIPFFT_Z2D, batch, &workSize) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftSetWorkArea(plan, nullptr) == HIPFFT_SUCCESS);
    EXPECT_TRUE(hipfftDestroy(plan) == HIPFFT_SUCCESS);
}

TEST(hipfftTest, RunR2C)
{
    const size_t N = 4096;
    float        in[N];
    for(size_t i = 0; i < N; i++)
        in[i] = i + (i % 3) - (i % 7);

    hipfftReal*    d_in;
    hipfftComplex* d_out;
    hipMalloc(&d_in, N * sizeof(hipfftReal));
    hipMalloc(&d_out, (N / 2 + 1) * sizeof(hipfftComplex));

    hipMemcpy(d_in, in, N * sizeof(hipfftReal), hipMemcpyHostToDevice);

    hipfftHandle plan;
    hipfftCreate(&plan);
    size_t workSize;
    hipfftMakePlan1d(plan, N, HIPFFT_R2C, 1, &workSize);

    hipfftExecR2C(plan, d_in, d_out);

    std::vector<hipfftComplex> out(N / 2 + 1);
    hipMemcpy(&out[0], d_out, (N / 2 + 1) * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    hipfftDestroy(plan);
    hipFree(d_in);
    hipFree(d_out);
    // NOTE: keep this condition for ease of changing n for ad-hoc tests
    //
    // cppcheck-suppress knownConditionTrueFalse
    if(N % 2 != 0)
    {
        EXPECT_TRUE(workSize != 0);
    }

    double ref_in[N];
    for(size_t i = 0; i < N; i++)
        ref_in[i] = in[i];

    fftw_complex* ref_out;
    fftw_plan     ref_p;

    ref_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    ref_p   = fftw_plan_dft_r2c_1d(N, ref_in, ref_out, FFTW_ESTIMATE);
    fftw_execute(ref_p);

    double maxv  = 0;
    double nrmse = 0; // normalized root mean square error
    for(size_t i = 0; i < (N / 2 + 1); i++)
    {
        // printf("element %d: FFTW result %f, %f; hipFFT result %f, %f \n", \
        //   (int)i, ref_out[i][0], ref_out[i][1], out[i].x, out[i].y);
        double dr = ref_out[i][0] - out[i].x;
        double di = ref_out[i][1] - out[i].y;
        maxv      = fabs(ref_out[i][0]) > maxv ? fabs(ref_out[i][0]) : maxv;
        maxv      = fabs(ref_out[i][1]) > maxv ? fabs(ref_out[i][1]) : maxv;

        nrmse += ((dr * dr) + (di * di));
    }
    nrmse /= (double)((N / 2 + 1));
    nrmse = sqrt(nrmse);
    nrmse /= maxv;

    EXPECT_TRUE(nrmse < type_epsilon<double>());
    fftw_free(ref_out);
}

// ask for a transform whose parameters are only valid out-of-place.
// since hipFFT generates both in-place and out-place plans up front
// (because it's not told about the placement until exec time), this
// ensures that a failure to create an in-place plan doesn't prevent
// the out-place plan from working.
TEST(hipfftTest, OutplaceOnly)
{
    int   N_in  = 4;
    int   N_out = N_in / 2 + 1;
    float in[N_in];
    for(int i = 0; i < N_in; i++)
        in[i] = i + (i % 3) - (i % 7);

    hipfftReal*    d_in;
    hipfftComplex* d_out;
    hipMalloc(&d_in, N_in * sizeof(hipfftReal));
    hipMalloc(&d_out, N_out * sizeof(hipfftComplex));

    hipMemcpy(d_in, in, N_in * sizeof(hipfftReal), hipMemcpyHostToDevice);

    hipfftHandle plan;
    ASSERT_EQ(hipfftCreate(&plan), HIPFFT_SUCCESS);

    ASSERT_EQ(hipfftPlanMany(&plan, 1, &N_in, &N_in, 1, N_in, &N_out, 1, N_out, HIPFFT_R2C, 1),
              HIPFFT_SUCCESS);

    ASSERT_EQ(hipfftExecR2C(plan, d_in, d_out), HIPFFT_SUCCESS);

    std::vector<hipfftComplex> out(N_out);
    hipMemcpy(out.data(), d_out, N_out * sizeof(hipfftComplex), hipMemcpyDeviceToHost);

    // in-place transform isn't really *supposed* to work - this
    // might or might not fail but we can at least check that it
    // doesn't blow up.
    hipfftExecR2C(plan, reinterpret_cast<hipfftReal*>(d_out), d_out);

    hipfftDestroy(plan);
    hipFree(d_in);
    hipFree(d_out);

    double ref_in[N_in];
    for(int i = 0; i < N_in; i++)
        ref_in[i] = in[i];

    fftw_complex* ref_out;
    fftw_plan     ref_p;

    ref_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N_out);
    ref_p   = fftw_plan_dft_r2c_1d(N_in, ref_in, ref_out, FFTW_ESTIMATE);
    fftw_execute(ref_p);

    double maxv  = 0;
    double nrmse = 0; // normalized root mean square error
    for(int i = 0; i < N_out; i++)
    {
        // printf("element %d: FFTW result %f, %f; hipFFT result %f, %f \n", \
        //   (int)i, ref_out[i][0], ref_out[i][1], out[i].x, out[i].y);
        double dr = ref_out[i][0] - out[i].x;
        double di = ref_out[i][1] - out[i].y;
        maxv      = fabs(ref_out[i][0]) > maxv ? fabs(ref_out[i][0]) : maxv;
        maxv      = fabs(ref_out[i][1]) > maxv ? fabs(ref_out[i][1]) : maxv;

        nrmse += ((dr * dr) + (di * di));
    }
    nrmse /= (double)(N_out);
    nrmse = sqrt(nrmse);
    nrmse /= maxv;

    ASSERT_TRUE(nrmse < type_epsilon<double>());
    fftw_free(ref_out);
}
