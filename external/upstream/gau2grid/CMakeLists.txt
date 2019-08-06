#find_package(gau2grid 1.2 CONFIG QUIET COMPONENTS gaussian)

#if(${gau2grid_FOUND})
#    get_property(_loc TARGET gau2grid::gg PROPERTY LOCATION)
#    message(STATUS "${Cyan}Found gau2grid${ColourReset}: ${_loc} (found version ${gau2grid_VERSION})")
#    add_library(gau2grid_external INTERFACE)  # dummy
#else()
    include(ExternalProject)
    message(STATUS "Suitable gau2grid could not be located, ${Magenta}Building gau2grid${ColourReset} instead.")
    ExternalProject_Add(gau2grid
        DEPENDS pybind11
        GIT_REPOSITORY https://github.com/dgasmith/gau2grid
        GIT_TAG 562ed3ccf7fdc76c5d309befef6075e2fc6613bf
        UPDATE_COMMAND ""
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_download
        SOURCE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_src
        BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_bin
        INSTALL_DIR  ${STAGED_INSTALL_PREFIX}/gau2grid
        TMP_DIR      ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_tmp
        STAMP_DIR    ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_stamp
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DENABLE_XHOST=${ENABLE_XHOST}
                   -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                   -DBUILD_FPIC=${BUILD_FPIC}
                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                   -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
                   -DMAX_AM=8
                   -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
                   -DENABLE_GENERIC=${ENABLE_GENERIC}
)

    set(gau2grid_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/gau2grid CACHE PATH "path to internally built gau2gridConfig.cmake" FORCE)
#endif()


#find_package(gau2grid 1.2 CONFIG QUIET COMPONENTS gaussian)

#if(${gau2grid_FOUND})
#    get_property(_loc TARGET gau2grid::gg PROPERTY LOCATION)
#    message(STATUS "${Cyan}Found gau2grid${ColourReset}: ${_loc} (found version ${gau2grid_VERSION})")
#    add_library(gau2grid_external INTERFACE)  # dummy
#else()
#    include(ExternalProject)
#    message(STATUS "Suitable gau2grid could not be located, ${Magenta}Building gau2grid${ColourReset} instead.")
#    ExternalProject_Add(gau2grid
#        DEPENDS pybind11
#        GIT_REPOSITORY https://github.com/dgasmith/gau2grid
#        GIT_TAG 562ed3ccf7fdc76c5d309befef6075e2fc6613bf
#        UPDATE_COMMAND ""
#        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_download
#        SOURCE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_src
#        BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_bin
#        INSTALL_DIR  ${STAGED_INSTALL_PREFIX}/gau2grid
#        TMP_DIR      ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_tmp
#        STAMP_DIR    ${CMAKE_CURRENT_BINARY_DIR}/gau2grid_stamp
#        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#                   -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                   -DENABLE_XHOST=${ENABLE_XHOST}
#                   -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
#                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
#                   -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
#                   -DBUILD_FPIC=${BUILD_FPIC}
#                   -DENABLE_GENERIC=${ENABLE_GENERIC}
#                   -DMAX_AM=${MAX_AM_ERI}
#                   -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
#                   -DENABLE_GENERIC=${ENABLE_GENERIC}
#        CMAKE_CACHE_ARGS -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
#)
#
#    set(gau2grid_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/gau2grid CACHE PATH "path to internally built gau2gridConfig.cmake" FORCE)
#endif()
