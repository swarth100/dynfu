# Build all dynfu source files into library dynfy
add_module_library(dynfu)

# set path to the terra solver file
target_compile_definitions(dynfu PUBLIC TERRA_SOLVER_FILE=${CMAKE_SOURCE_DIR}/include/dynfu/utils/terra/energy.t)

# Link to dynfu library
target_link_libraries(dynfu
    kfusion
    gcov
    Threads::Threads
    ${Boost_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${OPT_LIBRARIES} -Wl,-rpath,${CUDA_HOME}/lib64 dl
    ${PCL_LIBRARIES}
    ${TERRA_LIBRARIES}
    ${OPENMESH_LIBRARIES}
)
