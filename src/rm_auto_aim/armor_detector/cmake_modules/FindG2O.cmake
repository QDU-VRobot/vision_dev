# Locate the g2o libraries A general framework for graph optimization.
#
# This module defines G2O_FOUND, if false, do not try to link against g2o
# G2O_LIBRARIES, path to the libg2o G2O_INCLUDE_DIR, where to find the g2o
# header files
#
# Niko Suenderhauf <niko@etit.tu-chemnitz.de> Adapted by Felix Endres
# <endres@informatik.uni-freiburg.de>

if(UNIX)

  # IF(G2O_INCLUDE_DIR AND G2O_LIBRARIES)
  # in cache already SET(G2O_FIND_QUIETLY TRUE) ENDIF(G2O_INCLUDE_DIR AND
  # G2O_LIBRARIES)

  message(STATUS "Searching for g2o ...")
  find_path(
    G2O_INCLUDE_DIR
    NAMES core math_groups types
    PATHS /usr/local /usr
    PATH_SUFFIXES include/g2o include)

  if(G2O_INCLUDE_DIR)
    message(STATUS "Found g2o headers in: ${G2O_INCLUDE_DIR}")
  endif()

  find_library(
    G2O_CORE_LIB
    NAMES g2o_core g2o_core_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_STUFF_LIB
    NAMES g2o_stuff g2o_stuff_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_TYPES_SLAM2D_LIB
    NAMES g2o_types_slam2d g2o_types_slam2d_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_TYPES_SLAM3D_LIB
    NAMES g2o_types_slam3d g2o_types_slam3d_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_SOLVER_CHOLMOD_LIB
    NAMES g2o_solver_cholmod g2o_solver_cholmod_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_SOLVER_PCG_LIB
    NAMES g2o_solver_pcg g2o_solver_pcg_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_SOLVER_CSPARSE_LIB
    NAMES g2o_solver_csparse g2o_solver_csparse_rd
    PATHS /usr/local /usr
    PATH_SUFFIXES lib)
  find_library(
    G2O_INCREMENTAL_LIB
    NAMES g2o_incremental g2o_incremental_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)
  find_library(
    G2O_CSPARSE_EXTENSION_LIB
    NAMES g2o_csparse_extension g2o_csparse_extension_rd
    PATHS /usr/local /usr ${CMAKE_PREFIX_PATH}
    PATH_SUFFIXES lib)

  set(G2O_LIBRARIES
      ${G2O_CSPARSE_EXTENSION_LIB}
      ${G2O_CORE_LIB}
      ${G2O_STUFF_LIB}
      ${G2O_TYPES_SLAM2D_LIB}
      ${G2O_TYPES_SLAM3D_LIB}
      ${G2O_SOLVER_CHOLMOD_LIB}
      ${G2O_SOLVER_PCG_LIB}
      ${G2O_SOLVER_CSPARSE_LIB}
      ${G2O_INCREMENTAL_LIB})

  if(G2O_LIBRARIES AND G2O_INCLUDE_DIR)
    set(G2O_FOUND "YES")
    if(NOT G2O_FIND_QUIETLY)
      message(STATUS "Found libg2o: ${G2O_LIBRARIES}")
    endif()
  else(G2O_LIBRARIES AND G2O_INCLUDE_DIR)
    if(NOT G2O_LIBRARIES)
      if(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find libg2o!")
      endif()
    endif()

    if(NOT G2O_INCLUDE_DIR)
      if(G2O_FIND_REQUIRED)
        message(FATAL_ERROR "Could not find g2o include directory!")
      endif()
    endif()
  endif()

endif(UNIX)
