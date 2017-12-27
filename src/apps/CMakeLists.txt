# Link source files to executables
add_application(app)

# Link to all libraries
target_link_libraries(app
    ceres
    dynfu
    kfusion
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${OPT_LIBRARIES} -Wl,-rpath,${CUDA_HOME}/lib64 dl
    ${PCL_LIBRARIES}
    ${TERRA_LIBRARIES}
    ${OPENMESH_LIBRARIES}
)
