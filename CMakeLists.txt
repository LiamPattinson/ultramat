cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME "ultramat")
set(VERBOSE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# CMakeLists.txt
# Cmake settings file for Ultramat. Currently only builds tests.
# To compile, call `cmake path/to/ultramat/`, then `make`

# Set include directories
include_directories("${CMAKE_SOURCE_DIR}/src/")

# Get all header files
file(GLOB_RECURSE CXX_INC "${CMAKE_SOURCE_DIR}/src/*.hpp")

# =========================================================
# Compile options
# =========================================================

# Set C++ standard, plus warning errors
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Werror -pedantic -g -fPIC -fmax-errors=5")

# Compilation type?
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# =========================================================
# Dependencies
# =========================================================

# OpenMP
# Shared memory parallelism on a single node.
# To be implemented later
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# MPI
# Process-level parallelism, scales to multinode systems
# To be implemented later
find_package(MPI REQUIRED)
if(MPI_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()

# HDF5
# Portable scientific data format
# To be implemented later
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
    include_directories(${HDF5_CXX_INCLUDE_DIRS})
endif()

# =========================================================
# Unit Testing
# =========================================================

# Note: The following copied from Google Test github page with minor alterations.
# ===========================================================

# Download and unpack googletest at configure time
configure_file(CMakeListsTests.txt googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

# Now simply link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)

# ===========================================================

# Enable unit testing and get all test files
enable_testing()
file(GLOB_RECURSE CXX_TEST_FILES "${CMAKE_SOURCE_DIR}/test/*.cpp")
# Add each test
foreach(TEST_SRC ${CXX_TEST_FILES})
    string(REPLACE ${CMAKE_SOURCE_DIR}/ "" TEST_FILE ${TEST_SRC})
    get_filename_component(TEST_DIR ${TEST_FILE} DIRECTORY)
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    set(TEST_NAME "${TEST_NAME}.exe")
    set(TEST_EXE ${TEST_DIR}/${TEST_NAME})
    set(TEST_CMD ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${TEST_EXE})
    add_executable( ${TEST_NAME} ${TEST_SRC})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_CMD} )
    set_target_properties( ${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_DIR})
    target_compile_options(${TEST_NAME} PRIVATE ${HDF5_CXX_DEFINITIONS} ${MPI_CXX_COMPILE_FLAGS})
    target_link_libraries(${TEST_NAME} gtest_main)
    target_link_libraries(${TEST_NAME} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    target_link_libraries(${TEST_NAME} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
endforeach(TEST_SRC)
