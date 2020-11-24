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
# ############################################################################

# The ROCm platform requires Ubuntu 16.04 or Fedora 24, which has cmake 3.5
cmake_minimum_required( VERSION 3.5 )

# We use C++14 features, this will add compile option: -std=c++14
set( CMAKE_CXX_STANDARD 14 )

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if( WIN32 )
  set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" )
else( )
  set( CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories" )
endif( )

# Workaround for missing llvm cmake files in /opt/rocm/lib/cmake. Remove when this is fixed
list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

project( hipfft LANGUAGES CXX )

# Build options
option(BUILD_SHARED_LIBS "Build ${PROJECT_NAME} as a shared library" ON)
option(BUILD_VERBOSE "Output additional build information" OFF)

set (BUILD_WITH_COMPILER "HOST-default" CACHE STRING INTERNAL
     "Build ${PROJECT_NAME} with compiler HIP-hcc, HIP-clang, HIP-nvcc, or jus the host default compiler, i.e. g++")
set (BUILD_WITH_LIB "ROCM" CACHE STRING
     "Build ${PROJECT_NAME} with ROCM or CUDA libraries" )

option(BUILD_CLIENTS_TESTS "Build ${PROJECT_NAME} tests (requires 3rd dependencies)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" OFF)

# Look for libamdhip64
find_library(
  LIBAMDHIP64_LIBRARY
  NAMES libamdhip64.so
  HINTS "${CMAKE_PREFIX_PATH}/lib" )
message("amdhip64: ${LIBAMDHIP64_LIBRARY}")

# Set internal BUILD_WITH_COMPILER.
# TODO: more validity check.
if( NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  set (BUILD_WITH_COMPILER "HOST-default")
  if( CMAKE_CXX_COMPILER_VERSION VERSION_LESS "5.4.0" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU" )
    message(FATAL_ERROR "Compiler is too old. Please use g++ 5.4.0 or newer")
  endif( )
else()
  if( $ENV{HIP_PLATFORM} MATCHES "nvcc" )
    set (BUILD_WITH_COMPILER "HIP-nvcc")
  else()
    if ($ENV{HIP_COMPILER} MATCHES "clang")
      set (BUILD_WITH_COMPILER "HIP-clang")
      if (NOT BUILD_WITH_LIB STREQUAL "ROCM")
        message( FATAL_ERROR "Detected HIP_COMPILER=clang, but BUILD_WITH_LIB is not ROCM!" )
      endif()
    else()
      set (BUILD_WITH_COMPILER "HIP-hcc")
      if (NOT BUILD_WITH_LIB STREQUAL "ROCM")
        message( FATAL_ERROR "Detected BUILD_WITH_COMPILER=hcc, but BUILD_WITH_LIB is not ROCM!" )
      endif()
    endif()
  endif()
endif()

string( TOUPPER "${BUILD_WITH_COMPILER}" BUILD_WITH_COMPILER )
string( TOUPPER "${BUILD_WITH_LIB}" BUILD_WITH_LIB )

if (BUILD_WITH_COMPILER STREQUAL "HIP-NVCC" )
  set (BUILD_WITH_LIB "CUDA") # This may change after rocFFT fully support on NV platform
  set( HIP_PLATFORM "nvcc" )

  set( CMAKE_C_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_C_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_CXX_COMPILE_OPTIONS_PIC "-Xcompiler ${CMAKE_CXX_COMPILE_OPTIONS_PIC}" )
  set( CMAKE_SHARED_LIBRARY_C_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_CXX_FLAGS "-Xlinker ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  set( CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Xlinker -soname," )
  set( CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Xlinker -soname," )

  set( CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_C_FLAG "-Xlinker -rpath," )
  set( CMAKE_EXECUTABLE_RUNTIME_CXX_FLAG "-Xlinker -rpath," )

  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY "-Xcompiler='${CMAKE_C_COMPILE_OPTIONS_VISIBILITY}'" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY "-Xcompiler='${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY}'" )
  set( CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler='${CMAKE_C_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}'" )
  set( CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-Xcompiler='${CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN}'" )

else()
  if (BUILD_WITH_COMPILER STREQUAL "HIP-HCC" OR BUILD_WITH_COMPILER STREQUAL "HIP-CLANG" )
    set( HIP_PLATFORM "hcc" )
    set( AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target" )
  endif()

  if (BUILD_WITH_COMPILER STREQUAL "HIP-CLANG" )
    set(HIP_COMPILER "clang") # config HIP runtime for hip-clang, must be before find_package(hip)
  endif()
endif()

# Show the actual compiler(internal option)
message(STATUS "BUILD_WITH_COMPILER = " ${BUILD_WITH_COMPILER})

# Dependencies
include(cmake/dependencies.cmake)

set ( VERSION_STRING "1.0.1" )
rocm_setup_version( VERSION ${VERSION_STRING} )

set( hipfft_SOVERSION 0.1 ) # This is incremented when the ABI to the library changes

add_subdirectory( library )

# force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Build clients of the library
if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS )
  include( clients/cmake/build-options.cmake )
  add_subdirectory( clients )
endif( )
