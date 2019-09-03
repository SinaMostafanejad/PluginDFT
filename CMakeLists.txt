#
# @BEGIN LICENSE
#
# mydft by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2016 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(mydft LANGUAGES CXX C)

set(mydft_URL "https://github.com/edeprince3/RDMinoles")
set(mydft_EMAIL "m_mostafanejad@hotmail.com")
set(mydft_LICENSE "GNU Lesser General Public License, version 3 (LGPL-3.0)")
set(mydfft_DESCRIPTION "Open-Source Quantum Chemistry")


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH /home/sina/MySoftwares/psi4_May_19/cmake)
include(custom_color_messages)

# As a Psi4 plugin, the mydft project assumes psi4 is installed on the host
find_package(psi4 1.0 REQUIRED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${HOME}/mydft" CACHE PATH "Install path" FORCE)
endif()
message(STATUS "mydft install: ${CMAKE_INSTALL_PREFIX}")


set(STAGED_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/stage)
message(STATUS "mydft staging: ${STAGED_INSTALL_PREFIX}")

#=================================  Options  ===================================
#include(option_with_print)

option(ENABLE_GENERIC "Enables mostly static linking of language libraries for shared library" OFF)
option(CMAKE_BUILD_TYPE "Build type (Release or Debug)" Release)
option(BUILD_FPIC "Compile static libraries with position independent code" ON)
option(CMAKE_INSTALL_MESSAGE "Specify verbosity of installation messages" LAZY)
option(mydft_CXX_STANDARD "Specify C++ standard for mydft" 14)
option(BUILD_SHARED_LIBS "Build internally built mydft add-on libraries as shared, not static" OFF)
option(ENABLE_XHOST "Enables processor-specific optimization (with MSVC, it enables AVX2 instructions)" ON)

#===============================  Main Project  ================================
#include(ExternalProject)
#ExternalProject_Add(libxc_external
#       SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src
#       CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${STAGED_INSTALL_PREFIX}
#                  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
#                  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
#                  -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
#                  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
#                  -DCMAKE_C_FLAGS=${CMAKE_C_FLAGS}
#                  -DCMAKE_CXX_STANDARD=${mydft_CXX_STANDARD}
#                  -DCMAKE_CXX_STANDARD_REQUIRED=ON
#                  -DCMAKE_CXX_EXTENSIONS=OFF
#                  -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
#                  -DCMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS}
#                  -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
#                  -DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}
#                  -DCMAKE_INSTALL_DATADIR=${CMAKE_INSTALL_DATADIR}
#                  -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
#                  -DCMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE}
#                  -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
#                  -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}
#                  -DPYTHON_LIBRARY=${PYTHON_LIBRARY}
#                  -DENABLE_OPENMP=${ENABLE_OPENMP}
#                  -Dgau2grid_DIR=${gau2grid_DIR}
#                  -DLibxc_DIR=${Libxc_DIR}
#                  -DFortran_ENABLED=${Fortran_ENABLED}
#                  -DOpenMP_LIBRARY_DIRS=${OpenMP_LIBRARY_DIRS}
#                  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#       CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:PATH=${CMAKE_PREFIX_PATH}
#       BUILD_ALWAYS 1
#)

#add_subdirectory(doc)
add_subdirectory(src)
add_subdirectory(external/upstream)

#include(FetchContent)
#
#message(STATUS "${Red}Fetching the external library libxc ...${ColourReset}")
#FetchContent_Declare(libxc
#                    GIT_REPOSITORY git@gitlab.com:libxc/libxc.git
#                    GIT_TAG 0ba5d3c9daf9513830b79224dc4eadce51c2947d
#)
#FetchContent_GetProperties(libxc)
#message(STATUS "${Red}Fetching is done. Now populating the cache variables...${ColourReset}")
#if(NOT libxc_POPULATED)	
#  FetchContent_Populate(libxc)
#  add_subdirectory(${libxc_SOURCE_DIR} ${libxc_BINARY_DIR})
#endif()
#message(STATUS "${Red}Populating the cache variables is done.${ColourReset}")
#
#
#
#message(STATUS "${Red}Fetching the external library gau2grid ...${ColourReset}")
#FetchContent_Declare(gau2grid
#                    GIT_REPOSITORY git@github.com:dgasmith/gau2grid.git
#                    GIT_TAG 562ed3ccf7fdc76c5d309befef6075e2fc6613bf
#)
#FetchContent_GetProperties(gau2grid)
#message(STATUS "${Red}Fetching is done. Now populating the cache variables...${ColourReset}")
#if(NOT gau2grid_POPULATED)
#  FetchContent_Populate(gau2grid)
#  add_subdirectory(${gau2grid_SOURCE_DIR} ${gau2grid_BINARY_DIR})
#endif()
#message(STATUS "${Red}Populating the cache variables is done.${ColourReset}")

#find_package(gau2grid 1.2 CONFIG REQUIRED)
#get_property(_locgg TARGET gau2grid::gg PROPERTY LOCATION)
#list(APPEND _addons ${_locgg})
#message(STATUS "${Cyan}Using gau2grid${ColourReset}: ${_locgg} (version ${gau2grid_VERSION})")
#message(STATUS "${Yellow}The gau2grid header is in:${ColourReset}: ${gau2grid_INCLUDE_DIRS}/gau2grid")

#find_package(Libxc 4.3.4 CONFIG REQUIRED)
#get_property(_locxc TARGET Libxc::xc PROPERTY LOCATION)
#list(APPEND _addons ${_locxc})
#message(STATUS "${Cyan}Using Libxc${ColourReset}: ${_locxc} (version ${Libxc_VERSION})")
