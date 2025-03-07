#include <torch/torch.h>

namespace prac {
namespace nn {

class Conv2d: public torch::nn::Module {
public:
    Conv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0, bool bias = true)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), bias_(bias) {

        weight_ = register_parameter("weight", torch::empty({out_channels, in_channels, kernel_size, kernel_size}));

        if (bias) {
            bias_tensor_ = register_parameter("bias", torch::empty(out_channels));
        }

        torch::nn::init::kaiming_uniform_(weight_, std::sqrt(5.0));

        if (bias) {
            double fan_in = in_channels * kernel_size * kernel_size;
            double bound = 1.0 / std::sqrt(fan_in);
            torch::nn::init::uniform_(bias_tensor_, -bound, bound);
        }
    }
    virtual ~Conv2d() = default;

    virtual torch::Tensor forward(const torch::Tensor& input) = 0;

    torch::Tensor GetWeight() const {
        return weight_;
    }

    void SetWeight(const torch::Tensor& weight) {
        weight_ = weight;
    }

protected:
    int64_t in_channels_;
    int64_t out_channels_;
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool bias_;

    torch::Tensor weight_;
    torch::Tensor bias_tensor_;
};

} // namespace nn
} // namespace prac
