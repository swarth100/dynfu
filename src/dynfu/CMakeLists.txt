# Build all dynfu source files into library dynfy
add_module_library(dynfu)

# PCL Package must be included at this stage for DockerFile
find_package(PCL 1.2 REQUIRED COMPONENTS io)

# Add PCL to include paths
include_directories(
  ${PCL_INCLUDE_DIRS}
)

target_compile_definitions(dynfu
  PUBLIC TERRA_SOLVER_FILE=${CMAKE_SOURCE_DIR}/include/dynfu/utils/terra/energy.t)

# Link to dynfu library
target_link_libraries(dynfu
    ceres
    kfusion
    gcov
    Threads::Threads
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OPT_LIBRARIES}
    ${TERRA_LIBRARIES}
    ${OPENMESH_LIBRARIES}
    ${OPT_LIBRARIES} -Wl,-rpath,${CUDA_HOME}/lib64 dl
)
