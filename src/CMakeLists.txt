# ---------------------------------------------------------------------------- #
# BUILD ALL SUBPROJECTS
# ---------------------------------------------------------------------------- #
add_subdirectory(kfusion)
add_subdirectory(dynfu)
add_subdirectory(apps)

# ---------------------------------------------------------------------------- #
# SET OUTPUT DIRECTORIES
# ---------------------------------------------------------------------------- #
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
