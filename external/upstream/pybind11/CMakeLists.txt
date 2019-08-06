#find_package(pybind11 2.2.3 CONFIG QUIET)
#
#if(${pybind11_FOUND})
#    message(STATUS "${Cyan}Found pybind11${ColourReset}: ${pybind11_INCLUDE_DIR} (found version ${pybind11_VERSION})")
#    add_library(pybind11_external INTERFACE)  # dummy
#else()
    include(ExternalProject)
    message(STATUS "Suitable pybind11 could not be located, ${Magenta}Building pybind11${ColourReset} instead.")
    ExternalProject_Add(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG c9d32a81f40ad540015814edf13b29980c63e39c
        UPDATE_COMMAND ""
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/pybind11_download
        SOURCE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/pybind11_src
        BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR}/pybind11_bin
        INSTALL_DIR  ${STAGED_INSTALL_PREFIX}/pybind11
        TMP_DIR      ${CMAKE_CURRENT_BINARY_DIR}/pybind11_tmp
        STAMP_DIR    ${CMAKE_CURRENT_BINARY_DIR}/pybind11_stamp
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  # ditto
                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                   -DCMAKE_CXX_STANDARD=${mydft_CXX_STANDARD}
                   -DPYBIND11_TEST=OFF)

    set(pybind11_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/pybind11 CACHE PATH "path to internally built pybind11Config.cmake" FORCE)
#endif()
