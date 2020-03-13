// Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights
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

#include <complex>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipfft.h>

int main()
{
    std::cout << "hipfft 2D double-precision real-to-complex transform\n";

    const size_t Nx = 4;
    const size_t Ny = 5;

    std::cout << "Nx: " << Nx << "\tNy " << Ny << std::endl;

    const size_t Nycomplex = Ny / 2 + 1;
    const size_t rstride   = Nycomplex * 2; // Ny for out-of-place

    const size_t real_bytes    = sizeof(double) * Nx * rstride;
    const size_t complex_bytes = 2 * sizeof(double) * Nx * Ny;

    double*    x;
    hipError_t rt;
    rt = hipMalloc(&x, real_bytes);
    assert(rt == HIP_SUCCESS);

    std::vector<double> rdata(Nx * rstride);
    // Inititalize the data
    for(size_t i = 0; i < Nx * rstride; i++)
    {
        rdata[i] = i;
    }

    rt = hipMemcpy(x, rdata.data(), real_bytes, hipMemcpyHostToDevice);
    assert(rt == HIP_SUCCESS);

    std::cout << "input:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Ny; j++)
        {
            auto pos = i * rstride + j;
            std::cout << rdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Create plan:
    hipfftResult rc   = HIPFFT_SUCCESS;
    hipfftHandle plan = NULL;
    rc                = hipfftCreate(&plan);
    assert(rc == HIPFFT_SUCCESS);
    rc = hipfftPlan2d(&plan, // plan handle
                      Nx, // transform length
                      Ny, // transform length
                      HIPFFT_D2Z); // transform type (HIPFFT_R2C for single-precision)
    assert(rc == HIPFFT_SUCCESS);

    // Execute plan:
    // hipfftExecD2Z: double precision, hipfftExecR2C: single-precision
    rc = hipfftExecD2Z(plan, x, (hipfftDoubleComplex*)x);
    assert(rc == HIPFFT_SUCCESS);

    // copy the output data to the host:
    std::vector<std::complex<double>> cdata(Nx * Ny);
    hipMemcpy(cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
    std::cout << "output:\n";
    for(size_t i = 0; i < Nx; i++)
    {
        for(size_t j = 0; j < Nycomplex; j++)
        {
            auto pos = i * Nycomplex + j;
            std::cout << cdata[pos] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    hipfftDestroy(plan);
    hipFree(x);

    return 0;
}
