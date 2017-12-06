# ---------------------------------------------------------------------------- #
# GOOGLE TEST DEPENDECIES
# ---------------------------------------------------------------------------- #

# Implementation heavily based upon:
# https://github.com/kaizouman/gtest-cmake-example

find_package(Threads REQUIRED)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/master.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    # Disable install step
    INSTALL_COMMAND ""
)

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)

# Set libgmock properties
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
    "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)


# ---------------------------------------------------------------------------- #
# CODE COVERAGE (GCOV) SETTINGS
# ---------------------------------------------------------------------------- #
# Lcov Documentation:
# http://ltp.sourceforge.net/coverage/lcov/lcov.1.php
# http://swarminglogic.com/jotting/2014_05_lcov

# Exclude certain directories
SET(COVERAGE_EXCLUDES
  'src/kfusion/*'
  'src/apps/*'
  'include/kfusion/*'
  'include/nanoflann/*'
  'cmake/*'
  'test/*'
  'scripts/*'
  '/usr/*'
  'core/*'
  '/tmp/*'
  '/vol/project/*'
  '/vol/cuda/*'
  'build/*'
)

# ---------------------------------------------------------------------------- #
# TEST INCLUDE DIRECTORIES
# ---------------------------------------------------------------------------- #
include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include"
                    ${CUDA_INCLUDE_DIRS}
                    ${OPENMESH_INCLUDE_DIRS}
                    ${OPT_INCLUDE_DIRS}
                    ${TERRA_INCLUDE_DIRS})

# ---------------------------------------------------------------------------- #
# CREATE TESTS
# ---------------------------------------------------------------------------- #
# Force test build
set(gtest_build_samples ON)

# Add quaternion Test
CREATE_TEST(dynfu_test)

# Link all libraries for testing
target_link_libraries(dynfu_test
  dynfu
  ceres
  ${Boost_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${PCL_LIBRARIES}
  ${OPT_LIBRARIES}
  ${TERRA_LIBRARIES}
  ${OPENMESH_LIBRARIES}
)

# Generate coverage commands
SETUP_TARGET_FOR_COVERAGE(NAME coverage EXECUTABLE dynfu_test)
