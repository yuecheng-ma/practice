#include <torch/torch.h>
#include "naive_conv2d.h"

namespace prac {
namespace nn {

torch::Tensor NaiveConv2d::forward(const torch::Tensor& input) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    int64_t out_channels = weight_.size(0);
    int64_t kernel_height = weight_.size(2);
    int64_t kernel_width = weight_.size(3);

    int64_t out_height = (in_height + 2 * padding_ - kernel_height) / stride_ + 1;
    int64_t out_width = (in_width + 2 * padding_ - kernel_width) / stride_ + 1;

    auto output = torch::zeros({batch_size, out_channels, out_height, out_width});

    auto input_a = input.accessor<float, 4>();
    auto weight_a = weight_.accessor<float, 4>();
    auto output_a = output.accessor<float, 4>();

    for(int64_t n = 0; n < batch_size; ++n) {
        for(int64_t oc = 0; oc < out_channels; ++oc) {
            for(int64_t oh = 0; oh < out_height; ++oh) {
                for(int64_t ow = 0; ow < out_width; ++ow) {
                    float sum = 0.0;

                    const int64_t h_start = oh * stride_ - padding_;
                    const int64_t w_start = ow * stride_ - padding_;
                    printf("h_start: %ld, w_start: %ld\n", h_start, w_start);

                    for(int64_t ic = 0; ic < in_channels; ++ic) {
                        for(int64_t kh = 0; kh < kernel_height; ++kh) {
                            for(int64_t kw = 0; kw < kernel_width; ++kw) {
                                const int64_t h = h_start + kh;
                                const int64_t w = w_start + kw;

                                if (h >= 0 && h < in_height && w >= 0 && w < in_width) {
                                    sum += input_a[n][ic][h][w] * weight_a[oc][ic][kh][kw];
                                }
                                printf("sum(%f) += input_a[%ld][%ld][%ld][%ld](%f) * weight_a[%ld][%ld][%ld][%ld](%f)\n", sum, n, ic, h, w, input_a[n][ic][h][w], oc, ic, kh, kw, weight_a[oc][ic][kh][kw]);
                            }
                        }
                    }
                    printf("output_a[%ld][%ld][%ld][%ld] = %f\n", n, oc, oh, ow, sum);
                    output_a[n][oc][oh][ow] = sum;
                }
            }
        }
    }

    if (bias_tensor_.defined()) {
        auto bias_a = bias_tensor_.accessor<float, 1>();
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t oc = 0; oc < out_channels; ++oc) {
                for (int64_t oh = 0; oh < out_height; ++oh) {
                    for (int64_t ow = 0; ow < out_width; ++ow) {
                        output_a[n][oc][oh][ow] += bias_a[oc];
                    }
                }
            }
        }
    }

    return output;
}

} // namespace nn
} // namespace prac
