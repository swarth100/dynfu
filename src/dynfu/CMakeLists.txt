# Build all dynfu source files into library dynfy
add_module_library(dynfu)

find_package(PCL 1.2 REQUIRED COMPONENTS io)
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${OPT_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
# Link to dynfu library
target_link_libraries(dynfu
    ceres
    kfusion
    gcov
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OPT_LIBRARIES}
    ${PCL_LIBRARIES}
    ${TERRA_LIBRARIES}
)
