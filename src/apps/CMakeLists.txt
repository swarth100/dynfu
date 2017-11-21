include_directories(${CMAKE_SOURCE_DIR}/kfusion/include)

file(GLOB demo_SRCS *.cpp)

add_executable(demo ${demo_SRCS})

#
target_link_libraries(demo
    dynfu
    kfusion
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${OpenCV_LIBS}
)

set_target_properties(demo PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${demo_SRCS} DESTINATION app COMPONENT main)

