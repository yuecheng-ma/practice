#include <naive_conv2d.h>
#include <bits/stdc++.h>

int main() {
    auto x = torch::arange(1, 17).view({1, 1, 4, 4}).to(torch::kFloat);
    std::cout << x.sum().item<float>() << std::endl;

    std::shared_ptr<prac::nn::Conv2d> conv2d = std::make_shared<prac::nn::NaiveConv2d>(1, 1, 3, 1, 1, false);

    {
        torch::NoGradGuard no_grad;
        conv2d->SetWeight(torch::ones_like(conv2d->GetWeight()));
    }

    auto output = conv2d->forward(x);
    std::cout << output << std::endl;
    std::cout << output.sizes() << std::endl;

    return 0;
}