# cmake_minimum_required(VERSION 3.18)
# project(CUDA_GEMM_DEMO LANGUAGES CXX CUDA)  # 必须包含CUDA语言

# # 全局设置
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # 添加子目录（关键步骤）
# add_subdirectory(gemm)  # 这会将gemm子目录包含进构建系统

# # 创建主可执行文件
# add_executable(main_program main.cpp)

# # 链接库（关键步骤）
# target_link_libraries(main_program
#     PRIVATE
#         cuda_gemm  # 链接gemm子目录生成的库
# )

# # 包含路径配置
# target_include_directories(main_program
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/gemm  # 让main.cpp能找到cuda_gemm.h
# )

cmake_minimum_required(VERSION 3.18)
project(CUDA_GEMM_DEMO LANGUAGES CXX CUDA)

# 添加CUDA架构支持
set(CMAKE_CUDA_ARCHITECTURES "75")  # 自动检测GPU架构

# 添加库搜索路径
link_directories(${CMAKE_BINARY_DIR}/gemm)

add_subdirectory(gemm)

add_executable(main_program main.cpp)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(main_program PRIVATE
    cuda_gemm
    CUDA::cudart  # 显式链接CUDA运行时
)

target_include_directories(main_program PRIVATE
    ${CMAKE_SOURCE_DIR}/gemm
)