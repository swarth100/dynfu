# Build all dynfu source files into library dynfy
add_module_library(dynfu)

find_package(PCL 1.2 REQUIRED COMPONENTS io)
include_directories(
  ${PCL_INCLUDE_DIRS}
)
# Link to dynfu library
target_link_libraries(dynfu
    kfusion
    gcov
    ${Boost_LIBRARIES}
    ${PCL_LIBRARIES}
)
