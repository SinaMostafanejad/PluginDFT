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
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(mydft LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(custom_color_messages)

find_package(psi4 1.0 REQUIRED)

find_package(gau2grid 1.2 CONFIG REQUIRED)
get_property(_locgg TARGET gau2grid::gg PROPERTY LOCATION)
list(APPEND _addons ${_locgg})
message(STATUS "${Cyan}Using gau2grid${ColourReset}: ${_locgg} (version ${gau2grid_VERSION})")
message(STATUS "${Yellow}The gau2grid header is in:${ColourReset}: ${gau2grid_INCLUDE_DIRS}/gau2grid")

find_package(Libxc 4.3.4 CONFIG REQUIRED)
get_property(_locxc TARGET Libxc::xc PROPERTY LOCATION)
list(APPEND _addons ${_locxc})
message(STATUS "${Cyan}Using Libxc${ColourReset}: ${_locxc} (version ${Libxc_VERSION})")

add_subdirectory(${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/build)
