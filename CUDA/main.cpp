#include "gemm/cuda_gemm.h"
#include <bits/stdc++.h>

void Print(const std::vector<float>& mat) {
    for(const auto& num: mat) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::vector<float> a{1, 2, 3, 4, 5, 6};    // 2*3
    std::vector<float> b{7, 8, 9, 10, 11, 12}; // 3*2
    std::vector<float> c(4, 0.0f); // 2*2

    cuda_gemm(a.data(), b.data(), c.data(), 2, 3, 2);
    Print(c);

    return 0;
}