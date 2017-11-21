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

install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${demo_SRCS} DESTINATION app COMPONENT main)

