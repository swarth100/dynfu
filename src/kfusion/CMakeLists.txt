# Build all kfusion source files into library kfusion
add_module_library(kfusion)

# Link to kfusion library
target_link_libraries(kfusion
  dynfu
  ${CUDA_CUDART_LIBRARY}
  ${OPT_LIBRARIES}
  ${TERRA_LIBRARIES}
  ${OpenCV_LIBS}
)
