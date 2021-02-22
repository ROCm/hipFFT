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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# #############################################################################

# Dependencies

# Git
find_package(Git REQUIRED)

# HIP
list(APPEND CMAKE_MODULE_PATH /opt/rocm/hip/cmake)
if(NOT BUILD_WITH_LIB STREQUAL "CUDA")
  find_package(hip REQUIRED)
else()
  find_package(HIP REQUIRED)
  list( APPEND HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" )
endif()

# Either rocfft or cufft is required
if(NOT BUILD_WITH_LIB STREQUAL "CUDA")
  find_package(rocfft REQUIRED)
else()
  find_package(CUDA REQUIRED)
endif()

# ROCm cmake package
set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
find_package( ROCM CONFIG QUIET PATHS /opt/rocm )
if( NOT ROCM_FOUND )
  set( rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download" )
  file( DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip )

  execute_process( COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )

  find_package( ROCM REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
endif( )

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
