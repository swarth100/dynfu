add_module_library(dynfu)

target_link_libraries(dynfu
    kfusion
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)
