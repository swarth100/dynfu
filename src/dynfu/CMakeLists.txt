# Build all dynfu source files into library dynfy
add_module_library(dynfu)

# Link to dynfu library
target_link_libraries(dynfu
    gcov
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)
