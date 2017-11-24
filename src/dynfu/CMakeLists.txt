add_module_library(dynfu)

target_link_libraries(dynfu
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)
