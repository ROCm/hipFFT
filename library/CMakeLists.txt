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


# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on
# all the time.
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# Print verbose compiler flags
if( BUILD_VERBOSE )
  include( ../cmake/verbose.cmake )
endif()

# Configure a header file to pass the hipFFT version
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipfft-version.h.in"
  "${PROJECT_BINARY_DIR}/include/hipfft-version.h"
  )

# Public hipFFT headers
set( hipfft_headers_public
  include/hipfft.h
  include/hipfftXt.h
  ${PROJECT_BINARY_DIR}/include/hipfft-version.h
  )

source_group( "Header Files\\Public" FILES ${hipfft_headers_public} )

include( GNUInstallDirs )

set( BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR} )
set( LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR} )
set( INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} )

# nvcc can not recognize shared libraray file name with suffix other than *.so when linking.
#if (NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC")
#  rocm_setsoversion(hipfft ${hipfft_SOVERSION})
#endif()

# Include sources
include( src/CMakeLists.txt )

# Create hipFFT library
add_library( hipfft ${hipfft_source} ${hipfft_headers_public} )
add_library( hip::hipfft ALIAS hipfft )

# Target compile definitions
if( BUILD_WITH_COMPILER STREQUAL "HOST-DEFAULT" )
  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    target_compile_definitions( hipfft PRIVATE __HIP_PLATFORM_HCC__ )
  elseif( BUILD_WITH_LIB STREQUAL "CUDA" )
    target_compile_definitions( hipfft PRIVATE __HIP_PLATFORM_NVCC__ )
  endif()
endif()

# Target include directories
target_include_directories( hipfft
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
  $<INSTALL_INTERFACE:include>
  )

if( ROCFFT_CALLBACKS_ENABLED )
  target_compile_definitions( hipfft PRIVATE ROCFFT_CALLBACKS_ENABLED )
endif()

if( BUILD_WITH_LIB STREQUAL "CUDA" )
  target_include_directories( hipfft PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}> )
endif()

# Target link libraries
if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
  target_link_libraries( hipfft PRIVATE roc::rocfft )
else()
  target_link_libraries( hipfft PRIVATE ${CUDA_cufft_LIBRARY} )
endif()

# Target properties
set_target_properties( hipfft PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_propertieS( hipfft PROPERTIES DEBUG_POSTFIX "-d" )

#TODO:
# hipcc(with nvcc backend) build has problem for share library visibility,
# need to figure out the reason and enable visibility "hidden" for nvcc eventually.
if( NOT BUILD_WITH_COMPILER STREQUAL "HIP-NVCC" )
  set_target_properties( hipfft
    PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )
endif()

# Generate export header
include( GenerateExportHeader )
generate_export_header( hipfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipfft-export.h )

# Packaing...
if( ROCM_FOUND )
  rocm_install_targets( TARGETS hipfft
    INCLUDE
    ${CMAKE_SOURCE_DIR}/library/include
    ${CMAKE_BINARY_DIR}/include
    PREFIX hipfft )
  rocm_export_targets( TARGETS hip::hipfft
    PREFIX hipfft
    DEPENDS PACKAGE hip
    NAMESPACE hip:: )
  rocm_install_symlink_subdir( hipfft )

  # During transition to standalone hipFFT repository, don't install
  # symlinks to hipFFT headers.  Once the hipFFT headers are removed
  # from rocFFT, we can delete this..
  if(NOT CMAKE_HOST_WIN32)
    install(CODE "
        file(REMOVE \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/include/hipfft.h)
        file(REMOVE \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/include/hipfft-version.h)
        file(REMOVE \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/include/hipfft-export.h)
      ")
  endif()

  # Package specific CPACK vars
  if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
    set( CPACK_DEBIAN_PACKAGE_DEPENDS "rocfft (>= 1.0.0)" )
    set( CPACK_RPM_PACKAGE_REQUIRES "rocfft >= 1.0.0" )
  else()
    set( CPACK_DEBIAN_PACKAGE_DEPENDS "cufft (>= 10.0.0)" )
    set( CPACK_RPM_PACKAGE_REQUIRES "cufft >= 10.0.0" )
  endif()

  set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md" )

  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
  endif()

  set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

  # Give hipfft compiled for CUDA backend a different name
  if( BUILD_WITH_LIB STREQUAL "ROCM" )
    set( package_name hipfft )
  else()
    set( package_name hipfft-alt )
  endif()

  set( HIPFFT_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

  rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "ROCm FFT marshalling library"
    MAINTAINER "hipfft-maintainer@amd.com"
    LDCONFIG
    LDCONFIG_DIR ${HIPFFT_CONFIG_DIR}
    )
endif()
