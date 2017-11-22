file(GLOB dynfu_SRCS *.cpp)

# Declare the library
#add_library(dynfu STATIC
    #${dynfu_SRCS}
#)

add_module_library(dynfu)

target_link_libraries(dynfu
    kfusion
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${OpenCV_LIBS}
)

# Specify here the include directories exported
# by this library
#target_include_directories(dynfu PUBLIC
    #${CMAKE_CURRENT_SOURCE_DIR}
#)
