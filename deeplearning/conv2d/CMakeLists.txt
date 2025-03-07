# 创建conv2d库
add_library(conv2d
    conv2d.h
    naive_conv2d.h
    navie_conv2d.cpp
)

# 包含目录
target_include_directories(conv2d PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

# 链接LibTorch
target_link_libraries(conv2d PUBLIC ${TORCH_LIBRARIES})

# 设置编译选项
target_compile_options(conv2d PRIVATE
    -Wall
    -Wextra
    -O2
)

# 如果是调试模式，添加调试信息
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(conv2d PRIVATE -g)
endif()