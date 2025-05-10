#include <torch/torch.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Tensor: " << tensor << std::endl;

    return 0;
}