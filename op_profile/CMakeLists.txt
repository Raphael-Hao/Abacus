cmake_minimum_required(VERSION 3.16)
project(MergeBatch CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LINK_LIBS "")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(CUDA)

list(APPEND LINK_LIBS cudart)

find_package(CUDNN)
list(APPEND LINK_LIBS ${CUDNN_LIBRARY})

# Switch off modern thread local for dmlc-core, please see:
# https://github.com/dmlc/dmlc-core/issues/571#issuecomment-543467484
add_definitions(-DDMLC_MODERN_THREAD_LOCAL=0)
# disable stack trace in exception by default.
add_definitions(-DDMLC_LOG_STACK_TRACE_SIZE=0)
include_directories(3rdparty/dmlc-core/include)


if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/cmake)
  add_subdirectory("3rdparty/dmlc-core")
  list(APPEND LINK_LIBS dmlc)
endif()


file(GLOB_RECURSE SOURCE "src/*.cc" "src/*.h" "include/*.h")
include_directories(include)
add_executable(profile_conv ${SOURCE})

target_link_libraries(profile_conv dmlc)
target_link_libraries(profile_conv ${LINK_LIBS})

message(STATUS "CMAKE_LINK_LIBRARIES: " "${LINK_LIBS}")
