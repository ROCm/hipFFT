
if (DEFINED ENV{ROCM_PATH})
  set(rocm_bin "$ENV{ROCM_PATH}/bin")
else()
  set(rocm_bin "/opt/rocm/bin")
endif()

set(CMAKE_CXX_COMPILER "${rocm_bin}/hipcc")
set(CMAKE_C_COMPILER "${rocm_bin}/hipcc")
