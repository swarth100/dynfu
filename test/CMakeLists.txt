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

# I couldn't make it work with INTERFACE_INCLUDE_DIRECTORIES
include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")

# ---------------------------------------------------------------------------- #
# CODE COVERAGE (GCOV) SETTINGS
# ---------------------------------------------------------------------------- #
# Add Flags for coverage report
APPEND_COVERAGE_COMPILER_FLAGS()

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
)

# ---------------------------------------------------------------------------- #
# CREATE TESTS
# ---------------------------------------------------------------------------- #
# Force test build
set(gtest_build_samples ON)

# Add quaternion Test
CREATE_TEST(quaternionTest quaternion_test.cpp)

SETUP_TARGET_FOR_COVERAGE(NAME "coverage" EXECUTABLE quaternionTest)

#CREATE_TEST(solverTest solver_test.cpp)
