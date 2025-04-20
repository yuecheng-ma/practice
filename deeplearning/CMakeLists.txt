# lib
set(DEEPLEARNING_LIBS
    conv2d
    # linear_regression
    # softmax_regression
)
foreach(lib ${DEEPLEARNING_LIBS})
    add_subdirectory(${lib})
endforeach()

# bin
foreach(lib ${DEEPLEARNING_LIBS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${lib}/test")
        # add_subdirectory(${lib}/test ${lib}_test EXCLUDE_FROM_ALL)
        add_subdirectory(${lib}/test ${lib}_test EXCLUDE_FROM_ALL)
    endif()
endforeach()