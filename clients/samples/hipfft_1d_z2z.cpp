// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.
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

#include <complex>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft.h>

int main()
{
    std::cout << "hipfft 1D double-precision complex-to-complex transform\n";

    const int Nx        = 8;
    int       direction = HIPFFT_FORWARD; // forward=-1, backward=1

    std::vector<std::complex<double>> cdata(Nx);
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();

    // Create HIP device object and copy data to device
    // Use hipfftComplex for single-precision
    hipError_t           rt;
    hipfftDoubleComplex* x;
    rt = hipMalloc(&x, complex_bytes);
    assert(rt == HIP_SUCCESS);

    // Inititalize the data
    for(size_t i = 0; i < Nx; i++)
    {
        cdata[i] = i;
    }
    hipMemcpy(x, cdata.data(), complex_bytes, hipMemcpyHostToDevice);
    std::cout << "input:\n";
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    // Create the plan
    hipfftHandle plan = NULL;
    hipfftResult rc   = HIPFFT_SUCCESS;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan1d(&plan, // plan handle
                      Nx, // transform length
                      HIPFFT_Z2Z, // transform type (HIPFFT_C2C for single-precision)
                      1); // number of transforms
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // hipfftExecZ2Z: double precision, hipfftExecC2C: for single-precision
    rc = hipfftExecZ2Z(plan, x, x, direction);
    assert(rc == HIPFFT_SUCCESS);

    std::cout << "output:\n";
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    for(size_t i = 0; i < cdata.size(); i++)
    {
        std::cout << cdata[i] << " ";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);

    return 0;
}
