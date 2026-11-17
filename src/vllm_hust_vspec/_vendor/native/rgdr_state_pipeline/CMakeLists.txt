cmake_minimum_required(VERSION 3.18)
project(vspec_rgdr_pipeline LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(SOC_VERSION "Ascend910B2" CACHE STRING "Pinned Ascend target")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Installed CANN SDK")
if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "Set ASCEND_HOME_PATH to the installed CANN SDK")
endif()
include("${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake")
find_package(Torch REQUIRED)
find_package(Python3 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
ascendc_library(vspec_rgdr_kernel SHARED "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp")
add_library(vspec_rgdr_adapter SHARED adapter.cpp)
set_target_properties(vspec_rgdr_adapter PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(vspec_rgdr_adapter PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
    "${TORCH_NPU_PATH}/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(vspec_rgdr_adapter PRIVATE "${TORCH_NPU_PATH}/lib" "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(vspec_rgdr_adapter PRIVATE ${TORCH_LIBRARIES} torch_npu ascendcl vspec_rgdr_kernel)
target_link_options(vspec_rgdr_adapter PRIVATE
    "-Wl,-rpath,$ORIGIN:${TORCH_NPU_PATH}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64")
