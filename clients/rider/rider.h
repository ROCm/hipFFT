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

#ifndef RIDER_H
#define RIDER_H

#include "hipfft.h"
#include <vector>

// This is used to either wrap a HIP function call, or to explicitly check a variable
// for an error condition.  If an error occurs, we throw.
// Note: std::runtime_error does not take unicode strings as input, so only strings
// supported
inline hipError_t
    hip_V_Throw(hipError_t res, const std::string& msg, size_t lineno, const std::string& fileName)
{
    if(res != hipSuccess)
    {
        std::stringstream tmp;
        tmp << "HIP_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
    return res;
}

inline hipfftResult lib_V_Throw(hipfftResult       res,
                                const std::string& msg,
                                size_t             lineno,
                                const std::string& fileName)
{
    if(res != HIPFFT_SUCCESS)
    {
        std::stringstream tmp;
        tmp << "LIB_V_THROWERROR< ";
        tmp << res;
        tmp << " > (";
        tmp << fileName;
        tmp << " Line: ";
        tmp << lineno;
        tmp << "): ";
        tmp << msg;
        std::string errorm(tmp.str());
        std::cout << errorm << std::endl;
        throw std::runtime_error(errorm);
    }
    return res;
}

#define HIP_V_THROW(_status, _message) hip_V_Throw(_status, _message, __LINE__, __FILE__)
#define LIB_V_THROW(_status, _message) lib_V_Throw(_status, _message, __LINE__, __FILE__)

std::vector<int> compute_stride(const std::vector<int>& length,
                                const std::vector<int>& stride0   = std::vector<int>(),
                                const bool              rcpadding = false)

{
    const int dim = length.size();

    std::vector<int> stride(dim);

    int dimoffset = 0;

    if(stride0.size() == 0)
    {
        // Set the contiguous stride:
        stride[dim - 1] = 1;
        dimoffset       = 1;
    }
    else
    {
        // Copy the input values to the end of the stride array:
        for(int i = 0; i < stride0.size(); ++i)
        {
            stride[dim - stride0.size() + i] = stride0[i];
        }
    }

    if(stride0.size() < dim)
    {
        // Compute any remaining values via recursion.
        for(int i = dim - dimoffset - stride0.size(); i-- > 0;)
        {
            auto lengthip1 = length[i + 1];
            if(rcpadding && i == dim - 2)
            {
                lengthip1 = 2 * (lengthip1 / 2 + 1);
            }
            stride[i] = stride[i + 1] * lengthip1;
        }
    }

    return stride;
}

// Check the input and output stride to make sure the values are valid for the transform.
// If strides are not set, load default values.
void check_set_iostride(const bool              inplace,
                        const bool              forward,
                        const hipfftType        transformType,
                        const std::vector<int>& length,
                        std::vector<int>&       istride,
                        std::vector<int>&       ostride)
{
    if(!istride.empty() && istride.size() != length.size())
    {
        throw std::runtime_error("Transform dimension doesn't match input stride length");
    }

    if(!ostride.empty() && ostride.size() != length.size())
    {
        throw std::runtime_error("Transform dimension doesn't match output stride length");
    }

    if(transformType == HIPFFT_Z2Z || transformType == HIPFFT_C2C)
    {
        // Complex-to-complex transform

        // User-specified strides must match for in-place transforms:
        if(inplace && !istride.empty() && !ostride.empty() && istride != ostride)
        {
            throw std::runtime_error("In-place transforms require istride == ostride");
        }

        // If the user only specified istride, use that for ostride for in-place
        // transforms.
        if(inplace && !istride.empty() && ostride.empty())
        {
            ostride = istride;
        }

        // If the strides are empty, we use contiguous data.
        if(istride.empty())
        {
            istride = compute_stride(length);
        }
        if(ostride.empty())
        {
            ostride = compute_stride(length);
        }
    }
    else
    {
        // Real/complex transform

        // Length of complex data
        auto clength = length;
        clength[0]   = length[0] / 2 + 1;

        if(inplace)
        {
            // Fastest index must be contiguous.
            if(!istride.empty() && istride[0] != 1)
            {
                throw std::runtime_error(
                    "In-place real/complex transforms require contiguous input data.");
            }
            if(!ostride.empty() && ostride[0] != 1)
            {
                throw std::runtime_error(
                    "In-place real/complex transforms require contiguous output data.");
            }
            if(!istride.empty() && !ostride.empty())
            {
                for(int i = 1; i < length.size(); ++i)
                {
                    if(forward && istride[i] != 2 * ostride[i])
                    {
                        throw std::runtime_error(
                            "In-place real-to-complex transforms strides are inconsistent.");
                    }
                    if(!forward && 2 * istride[i] != ostride[i])
                    {
                        throw std::runtime_error(
                            "In-place complex-to-real transforms strides are inconsistent.");
                    }
                }
            }
        }

        if(istride.empty())
        {
            if(forward)
            {
                // real data
                istride = compute_stride(length, {inplace ? clength[0] * 2 : 0});
            }
            else
            {
                // complex data
                istride = compute_stride(clength);
            }
        }

        if(ostride.empty())
        {
            if(forward)
            {
                // complex data
                ostride = compute_stride(clength);
            }
            else
            {
                // real data
                ostride = compute_stride(length, {inplace ? clength[0] * 2 : 0});
            }
        }
    }
    // Final validation:
    if(istride.size() != length.size())
    {
        throw std::runtime_error("Setup failed; inconsistent istride and length.");
    }
    if(ostride.size() != length.size())
    {
        throw std::runtime_error("Setup failed; inconsistent ostride and length.");
    }
}

#endif // RIDER_H
