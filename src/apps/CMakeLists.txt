file(GLOB app_SRCS *.cpp)

# Link source files to executables
add_executable(app ${app_SRCS})

# Link to all libraries
target_link_libraries(app
    dynfu
    kfusion
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${OpenCV_LIBS}
)

# Set the output paths correctly
default_properties(app)

