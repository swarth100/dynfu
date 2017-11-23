# ---------------------------------------------------------------------------- #
# CMAKE INIT CONFIGURATIONS
# ---------------------------------------------------------------------------- #
cmake_minimum_required(VERSION 2.8.8)

# ---------------------------------------------------------------------------- #
# SET THE PROJECT
# ---------------------------------------------------------------------------- #
project(kfusion C CXX)

# ---------------------------------------------------------------------------- #
# UTILITY FUNCTIONS (MACROS) AND PATHS
# ---------------------------------------------------------------------------- #
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
include(Utils)
include(Targets)
include(CodeCoverage)

# ---------------------------------------------------------------------------- #
# CMAKE SETTINGS
# ---------------------------------------------------------------------------- #
# Use both Debug and Release versions
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Make debug directory for output files
FILE(MAKE_DIRECTORY "files")

# Set CXX to compile with C++11 standard
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")

# Add Flags for coverage report
APPEND_COVERAGE_COMPILER_FLAGS()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)

# ---------------------------------------------------------------------------- #
# DEPENDENCIES
# ---------------------------------------------------------------------------- #
find_package(OpenCV REQUIRED COMPONENTS core viz highgui calib3d)
find_package(Ceres REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost 1.36.0 REQUIRED COMPONENTS filesystem)

# ---------------------------------------------------------------------------- #
# MISCELLANEOUS SETTINGS
# ---------------------------------------------------------------------------- #
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---------------------------------------------------------------------------- #
# CUDA SETTINGS
# ---------------------------------------------------------------------------- #
set(HAVE_CUDA 1)
#list(APPEND CUDA_NVCC_FLAGS "-arch;compute_20")
list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50")

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

# ---------------------------------------------------------------------------- #
# SET INCLUDE PATHS
# ---------------------------------------------------------------------------- #
include_directories(include
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------- #
# BUILD SRC DIRECTORY
# ---------------------------------------------------------------------------- #
add_subdirectory(src)

# ---------------------------------------------------------------------------- #
# BUILD TESTS
# ---------------------------------------------------------------------------- #
enable_testing()
add_subdirectory(test)
