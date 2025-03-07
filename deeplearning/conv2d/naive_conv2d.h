#include <torch/torch.h>
#include "conv2d.h"

namespace prac {
namespace nn {

class NaiveConv2d: public Conv2d  {
public:
    NaiveConv2d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride = 1, int64_t padding = 0, bool bias = true)
    : Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias) {};
    virtual ~NaiveConv2d() = default;

    torch::Tensor forward(const torch::Tensor& input) override;
};

} // namespace nn
} // namespace prac
