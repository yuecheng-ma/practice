cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deeplearning)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置LibTorch路径
set(TORCH_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch-cxx11-abi-shared-with-deps-2.6.0+cpu/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})

# 寻找Torch包
find_package(Torch REQUIRED)

# 添加子目录
add_subdirectory(conv2d)

# 主程序
add_executable(${PROJECT_NAME} main.cpp)

# 链接依赖库
target_link_libraries(${PROJECT_NAME} PRIVATE conv2d)

# 设置rpath，确保可执行文件能找到动态库
set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "${TORCH_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)