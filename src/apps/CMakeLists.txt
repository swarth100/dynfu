# Link source files to executables
add_application(app)

# Link to all libraries
target_link_libraries(app
    dynfu
    kfusion
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
