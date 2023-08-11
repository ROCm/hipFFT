# #############################################################################
# Copyright (C) 2020 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

find_package( Boost COMPONENTS program_options REQUIRED)
set( Boost_USE_STATIC_LIBS OFF )


set( hipfft_bench_source bench.cpp ../rocFFT/shared/array_validator.cpp )
set( hipfft_bench_includes bench.h ../rocFFT/shared/array_validator.h )

add_executable( hipfft-bench ${hipfft_bench_source} ${hipfft_bench_includes} )

target_compile_options( hipfft-bench PRIVATE ${WARNING_FLAGS} )

set_target_properties( hipfft-bench PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON )

target_include_directories( hipfft-bench
  PRIVATE
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../rocFFT/library/include>
  )
  
if( NOT CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  if( NOT BUILD_WITH_LIB STREQUAL "CUDA" )
    if( WIN32 )
      find_package( HIP CONFIG REQUIRED )
    else()
      find_package( HIP MODULE REQUIRED )
    endif()

    target_link_libraries( hipfft-bench PRIVATE hip::host hip::device )
  else()
    target_compile_definitions( hipfft-bench PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_include_directories( hipfft-bench PRIVATE ${HIP_INCLUDE_DIRS})
  endif()
else()
  if( BUILD_WITH_LIB STREQUAL "CUDA" AND DEFINED boost_program_options_VERSION )
    # NVCC doesn't like linking with files that don't end in .so, so
    # we add a hack to remove the version number as the suffix.
    string(REGEX REPLACE
      \.${boost_program_options_VERSION} ""
      Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE
      ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE})
  endif()
endif()

if ( BUILD_WITH_LIB STREQUAL "CUDA" )
  target_compile_options( hipfft-bench PRIVATE -arch sm_53 -gencode=arch=compute_53,code=sm_53 -Xptxas=-w)
  target_link_libraries( hipfft-bench PRIVATE ${CUDA_LIBRARIES} )
else()
  if( NOT hiprand_FOUND )
    find_package( hiprand REQUIRED )
  endif()
  target_link_libraries( hipfft-bench PRIVATE hip::hiprand )
endif()

target_link_libraries( hipfft-bench PRIVATE hip::hipfft ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE} )

set_target_properties( hipfft-bench PROPERTIES CXX_EXTENSIONS NO )
set_target_properties( hipfft-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

rocm_install(TARGETS hipfft-bench COMPONENT benchmarks)

# install compatibility for old name of bench program - symlink on
# unix, hardlink on windows (since privilege is required to create
# symlinks there)
if( WIN32 )
  set( BENCH_LINK_COMMAND create_hardlink )
  set( BENCH_NEW_NAME ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/hipfft-bench${CMAKE_EXECUTABLE_SUFFIX} )
  set( BENCH_OLD_NAME ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/hipfft-rider${CMAKE_EXECUTABLE_SUFFIX} )
else()
  set( BENCH_LINK_COMMAND create_symlink )
  set( BENCH_NEW_NAME hipfft-bench )
  set( BENCH_OLD_NAME ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/hipfft-rider )
endif()
install(
  CODE "execute_process( COMMAND \"${CMAKE_COMMAND}\" -E ${BENCH_LINK_COMMAND} \"${BENCH_NEW_NAME}\" \"${BENCH_OLD_NAME}\" )"
)
