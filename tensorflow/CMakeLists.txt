# Add this directory for internal users.
include_directories(.)
# Find sources.
file(GLOB metis_tf_sources *.cc)
# Build libmetis.
add_library(metis_tf ${METIS_LIBRARY_TYPE} ${GKlib_sources} ${metis_sources} ${metis_tf_sources})
if(UNIX)
  target_link_libraries(metis_tf mtf)
endif()

if(METIS_INSTALL)
  install(TARGETS metis_tf
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()
