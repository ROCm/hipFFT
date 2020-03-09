# #############################################################################
# Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
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

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_USE_STATIC_LIBS OFF )

# ########################################################################
# The following check are hacks to get compiles working for CUDA backend
# These should be removed in the future.

# If hipcc is set as compiler, we are compiling for CUDA backend and
# nvcc complains of illegal -dumpversion flag because cmake thinks nvcc is a GNU compiler
# A workaround is setting boost compiler explicitely

# NVCC can not compile fftw.3 header file appropriately before v3.3.5
# https://github.com/FFTW/fftw3/issues/18
# v3.3.5 is installed by default on Ubuntu 16, workaround is to trick fftw3
# into thinking nvcc is an Intel compiler as desribed in above issue
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  set( Boost_COMPILER "" )
  set( BOOST_LIBRARYDIR "/usr/lib/x86_64-linux-gnu/" )  # Don't know why we need this yet
endif( )
# ########################################################################

if(NOT Boost_LIBRARIES)
  # Fixes a bug in Boost's CMAKE where Boost_LIBRARIES is not set.
  set(Boost_LIBRARIES Boost::program_options)
endif()


find_package( GTest REQUIRED )
find_package( FFTW 3.0 REQUIRED MODULE COMPONENTS FLOAT DOUBLE )

set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

set(hipfft_test_source
  gtest_main.cpp
  simple_test.cpp
  )

add_executable( hipfft-test ${hipfft_test_source} ${hipfft_test_includes} )

# Target compile definitions
if(BUILD_WITH_COMPILER STREQUAL "HOST-DEFAULT")
   if (BUILD_WITH_LIB STREQUAL "ROCM")
     target_compile_definitions(hipfft-test PRIVATE __HIP_PLATFORM_HCC__)
   elseif(BUILD_WITH_LIB STREQUAL "CUDA")
     target_compile_definitions(hipfft-test PRIVATE __HIP_PLATFORM_NVCC__)
   endif()
endif()

# Target compile features
if(NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC")
  target_compile_features( hipfft-test PRIVATE cxx_static_assert cxx_nullptr cxx_auto_type )
endif()

# Target include directories
target_include_directories( hipfft-test
  PRIVATE
    $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${FFTW_INCLUDES}>
    $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
)

target_link_libraries( hipfft-test PRIVATE
  roc::hipfft
  ${GTEST_LIBRARIES}
  ${FFTW_LIBRARIES}
  ${Boost_LIBRARIES}
  Threads::Threads
)

if (BUILD_WITH_LIB STREQUAL "CUDA")
  target_link_libraries( hipfft-test PRIVATE ${CUDA_LIBRARIES} )
endif()

if(BUILD_WITH_COMPILER STREQUAL "HOST-DEFAULT" OR BUILD_WITH_LIB STREQUAL "CUDA")
  get_target_property(HIPHCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE)
  target_link_libraries(hipfft-test PRIVATE ${HIPHCC_LOCATION})
endif()

set_target_properties( hipfft-test PROPERTIES DEBUG_POSTFIX "-d" CXX_EXTENSIONS NO )
set_target_properties( hipfft-test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
