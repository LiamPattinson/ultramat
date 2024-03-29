# CMakeLists.txt
# Cmake settings file for Ultramat unit testing.
# To build, call `cmake .` from this directory, followed by 'cmake --build .'
# To build in debug mode, call 'cmake . -DCMAKE_BUILD_TYPE=DEBUG' first.
# It is recommended to run tests with the command 'ctest --verbose' from the unit_testing directory

cmake_minimum_required(VERSION 3.5)
project("ultramat_unit_tests")
set(VERBOSE ON)

# Compilation type?
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C++ standard, plus warning errors. Export compile commands to a json
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -fPIC")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("Debug mode.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()

# Dependencies
# =========================================================

# Ultramat
find_package(ultramat CONFIG REQUIRED)

# OpenMP
# Shared memory parallelism on a single node.
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# MPI
# Process-level parallelism, scales to multinode systems
# To be implemented later
if(ULTRAMAT_USE_MPI)
    find_package(MPI REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
    include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
endif()

# HDF5
# Portable scientific data format
# To be implemented later
if(ULTRAMAT_USE_HDF5)
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5 REQUIRED COMPONENTS CXX)
    include_directories(${HDF5_INCLUDE_DIRS})
    include_directories(${HDF5_CXX_INCLUDE_DIRS})
endif()

# Google test
# Note: The following was copied from Google Test github page with minor alterations.
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

# Now simply link against gtest or gtest_main as needed. Eg
#add_executable(example example.cpp)
#target_link_libraries(example gtest_main)
#add_test(NAME example_test COMMAND example)


# Enable unit testing and get all test files
# ===========================================================

enable_testing()
file(GLOB_RECURSE CXX_TEST_FILES "${${PROJECT_NAME}_SOURCE_DIR}/src/*.cpp")
set(TEST_DIR "${${PROJECT_NAME}_SOURCE_DIR}/bin")
# Add each test
foreach(TEST_SRC ${CXX_TEST_FILES})
    string(REPLACE ${${PROJECT_NAME}_SOURCE_DIR}/ "" TEST_FILE ${TEST_SRC})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    set(TEST_NAME "${TEST_NAME}.exe")
    set(TEST_EXE ${TEST_DIR}/${TEST_NAME})
    if(USE_MPI)
        set(TEST_CMD ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${TEST_EXE})
    else()
        set(TEST_CMD ${TEST_EXE})
    endif()
    add_executable( ${TEST_NAME} ${TEST_SRC})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_CMD} )
    set_target_properties( ${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TEST_DIR})
    target_link_libraries(${TEST_NAME} ultramat::ultramat)
    target_link_libraries(${TEST_NAME} gtest_main)
    if(USE_MPI)
        target_link_libraries(${TEST_NAME} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
        target_compile_options(${TEST_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()
    if(USE_HDF5)
        target_compile_options(${TEST_NAME} PRIVATE ${HDF5_CXX_DEFINITIONS})
        target_link_libraries(${TEST_NAME} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
    endif()
endforeach(TEST_SRC)
