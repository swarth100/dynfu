# Build all kfusion source files into library kfusion
add_module_library(kfusion)

ADD_PCL()

# Link to kfusion library
target_link_libraries(kfusion
  dynfu
  ${CUDA_CUDART_LIBRARY}
  ${OpenCV_LIBS}
)
