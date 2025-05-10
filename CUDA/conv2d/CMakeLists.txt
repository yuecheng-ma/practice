add_library(conv2d SHARED
    naive_conv2d.cpp
    naive_conv2d.h
    conv2d.h
)

target_link_libraries(conv2d
    PUBLIC
    ${TORCH_LIBRARIES}
)

target_include_directories(conv2d
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)