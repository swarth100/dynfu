# Link source files to executables
add_application(app)

find_package(PCL 1.2 REQUIRED COMPONENTS io)

include_directories(
  ${PCL_INCLUDE_DIRS}
)

# Link to all libraries
target_link_libraries(app
    ceres
    dynfu
    kfusion
    ${Boost_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${OpenCV_LIBS}
    ${TERRA_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OPENMESH_LIBRARIES}
    ${OPT_LIBRARIES} -Wl,-rpath,${CUDA_HOME}/lib64 dl
)
