# Cmake settings file for Ultramat. Installs header-only library.
#
# To build, first hop in the 'build' directory and call "cmake ..".
# You can specify where to build with the flag "-DCMAKE_INSTALL_PREFIX:PATH=/your/installation/path"
# (You could do this from the root directory of this project, but it'll make a mess)
#
# To install, then call "sudo cmake --build . --target install".
# You'll need administrative privileges to do this.
#
# Massive thanks to Dominik Berner for the helpful cmake guide:  "dominikberner.ch/cmake-interface-lib/"
# TODO figure out how to set up variables ultramat_CXX_FLAGS and ultramat_EXE_LINKER_FLAGS

cmake_minimum_required(VERSION 3.5)
project("ultramat"
    VERSION 0.0.0 # proper versioning will begin once matrices exist...
    DESCRIPTION "Yet another header-only C++ linear algebra library."
    HOMEPAGE_URL "https://github.com/LiamPattinson/ultramat" 
)

# General config bits
set(VERBOSE ON)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Define interface library
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Set requirements for the build
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# Set install instructions
install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}_Targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Set versioning info
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION
    ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

# Set targets to install
install(EXPORT ${PROJECT_NAME}_Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
)

install(FILES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
