target_sources(mydft
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/dft.cc
               ${CMAKE_CURRENT_LIST_DIR}/dft.h
)

target_include_directories(mydft PRIVATE ${CMAKE_CURRENT_LIST_DIR})
