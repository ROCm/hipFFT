# #############################################################################
# Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_USE_STATIC_LIBS OFF )
if(NOT Boost_LIBRARIES)
  # Fixes a bug in Boost's CMAKE where Boost_LIBRARIES is not set.
  set(Boost_LIBRARIES Boost::program_options)
endif()

set(hipfft_rider_source rider.cpp )
set( hipfft_rider_includes rider.h )

add_executable( hipfft-rider ${hipfft_rider_source} ${hipfft_rider_includes} )

# Target compile definitions
if(BUILD_WITH_COMPILER STREQUAL "HOST-DEFAULT")
   if (BUILD_WITH_LIB STREQUAL "ROCM")
     target_compile_definitions(hipfft-rider PRIVATE __HIP_PLATFORM_HCC__)
   elseif(BUILD_WITH_LIB STREQUAL "CUDA")
     target_compile_definitions(hipfft-rider PRIVATE __HIP_PLATFORM_NVCC__)
   endif()
endif()

target_include_directories( hipfft-rider
  PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)


if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  target_link_libraries( hipfft-rider PRIVATE hip::hipfft ${Boost_LIBRARIES})
else()
  target_link_libraries( hipfft-rider PRIVATE hip::host hip::hipfft ${Boost_LIBRARIES})
endif()

if (BUILD_WITH_LIB STREQUAL "CUDA")
  target_link_libraries( hipfft-rider PRIVATE ${CUDA_LIBRARIES} )
endif()

set_target_properties( hipfft-rider PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipfft-rider
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
