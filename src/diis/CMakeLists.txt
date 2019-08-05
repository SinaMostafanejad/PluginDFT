target_sources(mydft
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/diis.cc
               ${CMAKE_CURRENT_SOURCE_DIR}/diis.h
)

target_include_directories(mydft PUBLIC ${CMAKE_CURRENT_LIST_DIR})
