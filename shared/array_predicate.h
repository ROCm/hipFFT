// Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_ARRAY_PREDICATE_H
#define ROCFFT_ARRAY_PREDICATE_H

#include "rocfft/rocfft.h"

namespace
{
    bool array_type_is_complex(rocfft_array_type type)
    {
        return type == rocfft_array_type_complex_interleaved
               || type == rocfft_array_type_complex_planar
               || type == rocfft_array_type_hermitian_interleaved
               || type == rocfft_array_type_hermitian_planar;
    }
    bool array_type_is_interleaved(rocfft_array_type type)
    {
        return type == rocfft_array_type_complex_interleaved
               || type == rocfft_array_type_hermitian_interleaved;
    }
    bool array_type_is_planar(rocfft_array_type type)
    {
        return type == rocfft_array_type_complex_planar
               || type == rocfft_array_type_hermitian_planar;
    }
}

#endif
