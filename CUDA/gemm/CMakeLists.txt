# cmake_minimum_required(VERSION 3.18)
# project(cuda_gemm LANGUAGES CXX CUDA)  # 必须声明CUDA为项目语言

# # 基础配置
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(THREADS_PREFER_PTHREAD_FLAG ON)

# # 共享库配置
# add_library(cuda_gemm SHARED
#     naive_gemm.cu
#     cuda_gemm.h
# )

# target_include_directories(cuda_gemm
#     PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#         $<INSTALL_INTERFACE:include>
# )

# # CUDA专用配置
# set_target_properties(cuda_gemm PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     POSITION_INDEPENDENT_CODE ON
# )

# # 查找CUDA依赖
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(cuda_gemm
#     PRIVATE
#         CUDA::cudart
# )

# # 可执行测试程序
# if(BUILD_TESTING)
#     enable_testing()
#     add_executable(test_gemm test/test_gemm.cpp)
#     target_link_libraries(test_gemm PRIVATE cuda_gemm)
#     add_test(NAME gemm_test COMMAND test_gemm)
# endif()

# # 安装规则
# install(TARGETS cuda_gemm
#     EXPORT cuda_gemm-targets
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(DIRECTORY include/
#     DESTINATION include
#     FILES_MATCHING PATTERN "*.h"
# )

# install(EXPORT cuda_gemm-targets
#     FILE cuda_gemm-config.cmake
#     DESTINATION lib/cmake/cuda_gemm
# )

# # 架构自动检测（可选）
# include(CheckLanguage)
# check_language(CUDA)
# if(CMAKE_CUDA_COMPILER)
#     set(CMAKE_CUDA_ARCHITECTURES "native")  # 自动检测本地GPU架构
#     message(STATUS "Found CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
# endif()

# 创建共享库目标
add_library(cuda_gemm SHARED
    naive_gemm.cu
    cuda_gemm.h
)

# CUDA专用配置
set_target_properties(cuda_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 包含目录（对外暴露）
target_include_directories(cuda_gemm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # 使主程序能找到头文件
)

# 链接CUDA运行时库
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_gemm
    PRIVATE
        CUDA::cudart
)

# 安装规则（可选）
install(TARGETS cuda_gemm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES cuda_gemm.h
    DESTINATION include
)