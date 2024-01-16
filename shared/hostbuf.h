// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_HOSTBUF_H
#define ROCFFT_HOSTBUF_H

#include "arithmetic.h"
#include <cstdlib>
#include <cstring>

#ifndef WIN32
#include <stdlib.h>
#include <sys/mman.h>
#endif

// Simple RAII class for host buffers.  T is the type of pointer that
// data() returns
template <class T = void>
class hostbuf_t
{
public:
    hostbuf_t() {}
    // buffers are movable but not copyable
    hostbuf_t(hostbuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
    }
    hostbuf_t& operator=(hostbuf_t&& other)
    {
        std::swap(buf, other.buf);
        std::swap(bsize, other.bsize);
        return *this;
    }
    hostbuf_t(const hostbuf_t&) = delete;
    hostbuf_t& operator=(const hostbuf_t&) = delete;

    ~hostbuf_t()
    {
        free();
    }

    void alloc(size_t size)
    {
        bsize = size;
        free();

        // we're aligning to multiples of 64 bytes, so round the
        // allocation size up to the nearest 64 to keep ASAN happy
        if(size % 64)
        {
            size += 64 - size % 64;
        }

        // FFTW requires aligned allocations to use faster SIMD instructions.
        // If enabling hugepages, align to 2 MiB. Otherwise, aligning to
        // 64 bytes is enough for AVX instructions up to AVX512.
#ifdef WIN32
        buf = _aligned_malloc(size, 64);
#else
        // On Linux, ask for hugepages to reduce TLB pressure and
        // improve performance.  Allocations need to be aligned to
        // the hugepage size, and rounded up to the next whole
        // hugepage.
        static const size_t TWO_MiB = 2 * 1024 * 1024;
        if(size >= TWO_MiB)
        {
            size_t rounded_size = DivRoundingUp(size, TWO_MiB) * TWO_MiB;
            buf                 = aligned_alloc(TWO_MiB, rounded_size);
            madvise(buf, rounded_size, MADV_HUGEPAGE);
        }
        else
            buf = aligned_alloc(64, size);
#endif
    }

    size_t size() const
    {
        return bsize;
    }

    void free()
    {
        if(buf != nullptr)
        {
#ifdef WIN32
            _aligned_free(buf);
#else
            std::free(buf);
#endif
            buf   = nullptr;
            bsize = 0;
        }
    }

    T* data() const
    {
        return static_cast<T*>(buf);
    }

    // Copy method
    hostbuf_t copy() const
    {
        hostbuf_t copy;
        copy.alloc(bsize);
        memcpy(copy.buf, buf, bsize);
        return copy;
    }

    // shrink the buffer to fit the new size
    void shrink(size_t new_size)
    {
        if(new_size > bsize)
            throw std::runtime_error("can't shrink hostbuf to larger size");
        // just pretend the buffer is now that size
        bsize = new_size;
    }

    // equality/bool tests
    bool operator==(std::nullptr_t n) const
    {
        return buf == n;
    }
    bool operator!=(std::nullptr_t n) const
    {
        return buf != n;
    }
    operator bool() const
    {
        return buf;
    }

private:
    // The host buffer
    void*  buf   = nullptr;
    size_t bsize = 0;
};

// default hostbuf that gives out void* pointers
typedef hostbuf_t<> hostbuf;
#endif
