# ---------------------------------------------------------------------------- #
# CMAKE INIT CONFIGURATIONS
# ---------------------------------------------------------------------------- #
cmake_minimum_required(VERSION 2.8.8)

# ---------------------------------------------------------------------------- #
# ADD OPT AND TERRA DEPNEDENCIES
# ---------------------------------------------------------------------------- #

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/terra)
    set(TERRA_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/terra/include)
    set(TERRA_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/terra/lib/libterra.a)
else()
    message(FATAL_ERROR "Please run setup script to install terra.")
endif()

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/Opt)
    set(OPT_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/Opt/examples/external/mLib/include ${CMAKE_CURRENT_BINARY_DIR}/Opt/examples/shared/ ${CMAKE_CURRENT_BINARY_DIR}/Opt/API/release/include)
    set(OPT_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/Opt/API/release/lib/libOptDev.a)
else()
    message(FATAL_ERROR "Please run setup script to install Opt.")
endif()

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
if(BUILD_TESTS)
  include(CodeCoverage)
endif()

# ---------------------------------------------------------------------------- #
# CMAKE SETTINGS
# ---------------------------------------------------------------------------- #
# Use both Debug and Release versions
SET(CMAKE_CONFIGURATION_TYPES "Debug;Release")

# Make debug directory for output files
FILE(MAKE_DIRECTORY "files")

# Set CXX to compile with C++11 standard
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -MP -MD -pthread -fpermissive")

if(BUILD_TESTS)
  # Add Flags for coverage report
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()

# ---------------------------------------------------------------------------- #
# DEPENDENCIES
# ---------------------------------------------------------------------------- #
find_package(Boost 1.36.0 REQUIRED COMPONENTS filesystem)
find_package(Ceres REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core viz highgui calib3d)
find_package(OpenMesh REQUIRED)
# find_package(PCL 1.8 REQUIRED COMPONENTS common features geometry io surface search)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------- #
# CUDA SETTINGS
# ---------------------------------------------------------------------------- #
SET(HAVE_CUDA 1)
LIST(APPEND CUDA_NVCC_FLAGS
  "-gencode;arch=compute_30,code=sm_30;
   -gencode;arch=compute_35,code=sm_35;
   -gencode;arch=compute_50,code=sm_50;
   -gencode;arch=compute_61,code=sm_61;
   --ftz=true;
   --prec-div=false;
   --prec-sqrt=false;
   --expt-relaxed-constexpr"
)

if(UNIX OR APPLE)
  LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

# ---------------------------------------------------------------------------- #
# SET INCLUDE PATHS
# ---------------------------------------------------------------------------- #
include_directories(include
  ${Boost_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${OPENMESH_INCLUDE_DIRS}
  ${OPT_INCLUDE_DIRS}
  ${TERRA_INCLUDE_DIRS}
)

# ---------------------------------------------------------------------------- #
# BUILD SRC DIRECTORY
# ---------------------------------------------------------------------------- #
add_subdirectory(src)

# ---------------------------------------------------------------------------- #
# BUILD TESTS
# ---------------------------------------------------------------------------- #
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
