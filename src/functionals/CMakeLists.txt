target_sources(mydft
               PRIVATE
               ${CMAKE_CURRENT_SOURCE_DIR}/Func_HC.cc
)

target_include_directories(mydft PUBLIC ${CMAKE_CURRENT_LIST_DIR})
