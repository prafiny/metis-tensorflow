# Add this directory for internal users.
include_directories(.)
include_directories(../GKlib)
include_directories(../libmetis)
# Find sources.
file(GLOB metis_tf_sources *.cc)
file(GLOB metis_sources ../libmetis/*.c)
file(GLOB GKlib_sources ../GKlib/*.c)
# Build libmetis.
add_library(metis_tf ${METIS_LIBRARY_TYPE} ${GKlib_sources} ${metis_sources} ${metis_tf_sources})
if(UNIX)
  target_link_libraries(metis_tf "/usr/local/lib/python3.5/dist-packages/tensorflow/libtensorflow_framework.so")
endif()

if(METIS_INSTALL)
  install(TARGETS metis_tf
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib)
endif()
