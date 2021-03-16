# CMakeLists.txt
# Cmake settings file for Ultramat performance testing.
# To build, call `cmake .` from this directory, followed by 'cmake --build .'
# To build in debug mode, call 'cmake . -DCMAKE_BUILD_TYPE=DEBUG' first.
# To run all, I suggest 'ctest --verbose'.

cmake_minimum_required(VERSION 3.5)
project("ultramat_performance_tests")
set(VERBOSE ON)

# Compilation type?
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C++ standard, plus warning errors
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic -fPIC -fmax-errors=5")

if(CMAKE_BUILD_TYPE MATCHES DEBUG)
    message("Debug mode.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Dependencies
# =========================================================

# Ultramat
find_package(ultramat CONFIG REQUIRED)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)

# OpenMP
# Shared memory parallelism on a single node.
# To be implemented later
if(ULTRAMAT_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

# Enable testing and get all test files
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
    target_link_libraries(${TEST_NAME} ${ARMADILLO_LIBRARIES})
    target_link_libraries(${TEST_NAME} Eigen3::Eigen)
    if(USE_MPI)
        target_link_libraries(${TEST_NAME} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
        target_compile_options(${TEST_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    endif()
    if(USE_HDF5)
        target_compile_options(${TEST_NAME} PRIVATE ${HDF5_CXX_DEFINITIONS})
        target_link_libraries(${TEST_NAME} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})
    endif()
endforeach(TEST_SRC)
