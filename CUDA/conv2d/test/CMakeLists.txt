add_executable(conv2d_test
    conv2d_test.cpp
)

target_link_libraries(conv2d_test
    PRIVATE
    conv2d
    ${TORCH_LIBRARIES}
)