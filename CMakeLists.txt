# ---------------------------------------------------------------------------- #
# CMAKE INIT CONFIGURATIONS
# ---------------------------------------------------------------------------- #
cmake_minimum_required(VERSION 2.8.8)

# ---------------------------------------------------------------------------- #
# SET THE PROJECT
# ---------------------------------------------------------------------------- #
project(kfusion C CXX)

# ---------------------------------------------------------------------------- #
# CMAKE SETTINGS
# ---------------------------------------------------------------------------- #
# Use both Debug and Release versions
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Make debug directory for output files
FILE(MAKE_DIRECTORY "files")

# Set CXX to compile with C++11 standard
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11")

# ---------------------------------------------------------------------------- #
# CUDA SETTINGS
# ---------------------------------------------------------------------------- #
SET(HAVE_CUDA 1)
LIST(APPEND CUDA_NVCC_FLAGS
  "-gencode;arch=compute_30,code=sm_30;
   -gencode;arch=compute_35,code=sm_35;
   -gencode;arch=compute_50,code=sm_50
   --ftz=true;
   --prec-div=false;
   --prec-sqrt=false"
 )

# Apple settings
if(UNIX OR APPLE)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

# ---------------------------------------------------------------------------- #
# UTILITY FUNCTIONS (MACROS) AND PATHS
# ---------------------------------------------------------------------------- #
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
include(Utils)
include(Targets)
include(CodeCoverage)

# ---------------------------------------------------------------------------- #
# DEPENDENCIES
# ---------------------------------------------------------------------------- #
find_package(OpenCV REQUIRED COMPONENTS core viz highgui calib3d)
find_package(Ceres REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost 1.36.0 COMPONENTS filesystem)
find_package(PCL 1.2 REQUIRED)

# ---------------------------------------------------------------------------- #
# SET INCLUDE PATHS
# ---------------------------------------------------------------------------- #
include_directories(include
  ${OpenCV_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
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
