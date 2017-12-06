# Build all dynfu source files into library dynfy
add_module_library(dynfu)

find_package(PCL 1.2 REQUIRED COMPONENTS io)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OPENMESH_INCLUDE_DIRS}
  ${TERRA_INCLUDE_DIRS}
  ${OPT_INCLUDE_DIRS}
)

# Link to dynfu library
target_link_libraries(dynfu
    ceres
    kfusion
    gcov
    Threads::Threads
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
    ${TERRA_LIBRARIES}
    ${OPENMESH_LIBRARIES}
    ${OPT_LIBRARIES} -Wl,-rpath,${CUDA_HOME}/lib64 dl
)
