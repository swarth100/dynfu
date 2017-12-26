# PCL Hardcoded Path.
# Works for now, a better fix must be found in the future
LIST(APPEND CUDA_NVCC_FLAGS
  "-I/vol/project/2017/362/g1736211/include/pcl-1.8"
)

# Build all kfusion source files into library kfusion
add_module_library(kfusion)

ADD_PCL()

# Link to kfusion library
target_link_libraries(kfusion
  dynfu
  ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)
