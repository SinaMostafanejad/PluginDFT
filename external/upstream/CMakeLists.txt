foreach(dir_idx pybind11
                libxc
                gau2grid
)
       add_subdirectory(${dir_idx})
endforeach()
