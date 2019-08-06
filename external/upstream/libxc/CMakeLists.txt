#include(FetchContent)
#
#message(STATUS "${Magenta}Fetching the external library libxc ...${ColourReset}")
#FetchContent_Declare(libxc
#                    GIT_REPOSITORY git@gitlab.com:libxc/libxc.git
#                    GIT_TAG 0ba5d3c9daf9513830b79224dc4eadce51c2947d
#)
#FetchContent_GetProperties(libxc)
#message(STATUS "${Magenta}Fetching is done. Now populating the cache variables...${ColourReset}")
#if(NOT libxc_POPULATED)        
#  FetchContent_Populate(libxc)
#  add_subdirectory(${libxc_SOURCE_DIR} ${libxc_BINARY_DIR})
#endif()
#message(STATUS "${Magenta}Populating the cache variables is done.${ColourReset}")
#
#find_package(Libxc 4.3.3 CONFIG QUIET)

#if(${Libxc_FOUND})
#    get_property(_loc TARGET Libxc::xc PROPERTY LOCATION)
#    message(STATUS "${Cyan}Found Libxc${ColourReset}: ${_loc} (found version ${Libxc_VERSION})")
#    add_library(libxc_external INTERFACE)  # dummy
#else()
    include(ExternalProject)
    message(STATUS "Suitable Libxc could not be located, ${Magenta}Building Libxc${ColourReset} instead.")
    ExternalProject_Add(libxc
        GIT_REPOSITORY https://gitlab.com/libxc/libxc.git
        GIT_TAG 0ba5d3c9daf9513830b79224dc4eadce51c2947d
        UPDATE_COMMAND ""
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/libxc_download
        SOURCE_DIR   ${CMAKE_CURRENT_BINARY_DIR}/libxc_src
        BINARY_DIR   ${CMAKE_CURRENT_BINARY_DIR}/libxc_bin
        INSTALL_DIR  ${STAGED_INSTALL_PREFIX}/libxc
        TMP_DIR      ${CMAKE_CURRENT_BINARY_DIR}/libxc_tmp
        STAMP_DIR    ${CMAKE_CURRENT_BINARY_DIR}/libxc_stamp
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                   -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
                   -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
                   -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
                   -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
                   -DBUILD_FPIC=${BUILD_FPIC}
                   -DBUILD_TESTING=OFF
)
    set(Libxc_DIR ${STAGED_INSTALL_PREFIX}/share/cmake/Libxc CACHE PATH "path to internally built LibxcConfig.cmake" FORCE)
#endif()
