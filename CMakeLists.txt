cmake_minimum_required(VERSION 2.8)
project(METIS LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_C_FLAGS_DEBUG "-O0 -g")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TF_CFLAGS} -std=c++11" )
SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${TF_LFLAGS}" )

set(GKLIB_PATH "GKlib" CACHE PATH "path to GKlib")
set(SHARED TRUE CACHE BOOL "build a shared library")

if(MSVC)
  set(METIS_INSTALL FALSE)
else()
  set(METIS_INSTALL TRUE)
endif()

# Configure libmetis library.
if(SHARED)
  set(METIS_LIBRARY_TYPE SHARED)
else()
  set(METIS_LIBRARY_TYPE STATIC)
endif(SHARED)

include(${GKLIB_PATH}/GKlibSystem.cmake)
# Add include directories.
include_directories(${GKLIB_PATH})
include_directories(include)
# Recursively look for CMakeLists.txt in subdirs.
# no add_subdirectory("include")
# no add_subdirectory("libmetis")
add_subdirectory("tensorflow")
# no add_subdirectory("programs")
