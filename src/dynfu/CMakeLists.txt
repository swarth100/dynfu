add_module_library(dynfu)

target_link_libraries(dynfu
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)
