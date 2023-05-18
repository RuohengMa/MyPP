/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/sparse/xpu/conv.h"

namespace phi {
namespace sparse {

/**
 * x: (N, D, H, W, C)
 * kernel: (D, H, W, C, OC)
 * out: (N, D, H, W, OC)
 **/
template <typename T, typename IntT = int>
void Conv3dCooXPUKernel(const XPUContext& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& kernel,
                        const std::vector<int>& paddings,
                        const std::vector<int>& dilations,
                        const std::vector<int>& strides,
                        const int groups UNUSED,
                        const bool subm,
                        const std::string& key,
                        SparseCooTensor* out,
                        DenseTensor* rulebook,
                        DenseTensor* counter) {
  // update padding and dilation
  // Currently, only support x.layout is NDHWC, groups = 1
  // if x.layout != NDHWC then transpose(x), transpose(weight)

  const auto& x_dims = x.dims();
  const auto& kernel_dims = kernel.dims();
  int kernel_size = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
  DDim out_dims = {1, 1, 1, 1, 1};
  std::vector<int> kernel_sizes(kernel_dims.size());
  for (int i = 0; i < kernel_dims.size(); i++) {
    kernel_sizes[i] = kernel_dims[i];
  }

  std::vector<int> subm_paddings(paddings), subm_strides(strides);
  if (subm) {
    // the out shape of subm_conv is same as input shape
    // reset the padding=kernel_size/2 and strides=1
    phi::funcs::sparse::ResetSubmKernelSizeAndStrides(
        kernel.dims(), &subm_paddings, &subm_strides);
  }

  phi::funcs::sparse::GetOutShape(
      x_dims, kernel_sizes, subm_paddings, dilations, subm_strides, &out_dims);
  const int in_channels = kernel_dims[3];
  const int out_channels = kernel_dims[4];

  // Second algorithm:
  // https://pdfs.semanticscholar.org/5125/a16039cabc6320c908a4764f32596e018ad3.pdf
  // 1. product rulebook
  DenseTensor h_counter, h_offsets;
  h_counter.Resize({kernel_size});
  h_offsets.Resize({kernel_size + 1});
  int* h_counter_ptr = dev_ctx.template HostAlloc<int>(&h_counter);
  int* h_offsets_ptr = dev_ctx.template HostAlloc<int>(&h_offsets);

  // DenseTensor* rulebook = nullptr;
  const IntT* rulebook_ptr = nullptr;
  int n = 0;
  bool need_product_rulebook = true;
  if (subm && !key.empty()) {
    rulebook_ptr = phi::funcs::sparse::PrepareSubm<T, IntT, XPUContext>(
        dev_ctx,
        x,
        key,
        out_dims,
        out,
        h_counter_ptr,
        h_offsets_ptr,
        &n,
        &need_product_rulebook);
  }
  if (need_product_rulebook) {
    DenseTensor tmp_rulebook;
    ProductRuleBook<T, XPUContext, IntT>(dev_ctx,
                                         x,
                                         kernel_sizes,
                                         subm_paddings,
                                         dilations,
                                         subm_strides,
                                         out_dims,
                                         subm,
                                         &tmp_rulebook,
                                         h_counter_ptr);

    UpdateRulebookAndOutIndex<T, XPUContext, IntT>(
        dev_ctx, x, kernel_size, out_channels, out_dims, &tmp_rulebook, out);
    n = tmp_rulebook.dims()[1];
    rulebook_ptr = tmp_rulebook.data<IntT>();

    phi::funcs::sparse::SaveToTable(
        dev_ctx, x, key, tmp_rulebook, h_counter, out, rulebook, counter);
  }
  // int n = rulebook->dims()[1];

  // 2. gather
  DenseTensorMeta in_features_meta(
      phi::DataType::FLOAT32, {n, in_channels}, DataLayout::NHWC);
  DenseTensorMeta out_features_meta(
      phi::DataType::FLOAT32, {n, out_channels}, DataLayout::NHWC);
  phi::DenseTensor in_features =
      phi::Empty(dev_ctx, std::move(in_features_meta));
  phi::DenseTensor out_features =
      phi::Empty(dev_ctx, std::move(out_features_meta));
  float* in_features_ptr = in_features.data<float>();
  float* out_features_ptr = out_features.data<float>();

  phi::DenseTensor x_values = phi::Cast<T, phi::XPUContext>(
      dev_ctx, x.values(), phi::DataType::FLOAT32);
  int r = xpu::gather(dev_ctx.x_context(),
                      x_values.data<float>(),
                      rulebook_ptr + n,
                      in_features_ptr,
                      phi::vectorize<int64_t>(x.values().dims()),
                      n,
                      0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");

  // 3. call gemm for every werght
  int offset = 0;
  for (int i = 0; i < kernel_size; i++) {
    h_offsets_ptr[i] = offset;
    offset += h_counter_ptr[i];
  }
  h_offsets_ptr[kernel_size] = offset;

  DenseTensor tmp_kernel =
      phi::Cast<T, XPUContext>(dev_ctx, kernel, phi::DataType::FLOAT32);
  const float* kernel_ptr = tmp_kernel.data<float>();
  for (int i = 0; i < kernel_size; i++) {
    if (h_counter_ptr[i] <= 0) {
      continue;
    }

    // call gemm: (n, in_channels) * (in_channels, out_channels)
    const int M = h_counter_ptr[i];
    const int K = in_channels;   // in_channels
    const int N = out_channels;  // out_channels
    float* tmp_in_ptr = in_features_ptr + h_offsets_ptr[i] * in_channels;
    const float* tmp_kernel_ptr = kernel_ptr + i * K * N;
    float* tmp_out_ptr = out_features_ptr + h_offsets_ptr[i] * out_channels;
    r = xpu::fc<float, float, float, float>(dev_ctx.x_context(),
                                            tmp_in_ptr,
                                            tmp_kernel_ptr,
                                            tmp_out_ptr,
                                            M,
                                            N,
                                            K,
                                            false,
                                            false,
                                            nullptr,
                                            nullptr,
                                            nullptr);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "fc");
  }

  // 4. scatter
  xpu::VectorParam<IntT> index{
      nullptr, n, const_cast<IntT*>(rulebook_ptr) + n * 2};
  DenseTensorMeta tmp_out_meta(
      phi::DataType::FLOAT32, out->values().dims(), out->values().layout());
  phi::DenseTensor tmp_out = phi::Empty(dev_ctx, std::move(tmp_out_meta));
  r = xpu::scatter(dev_ctx.x_context(),
                   tmp_out.data<float>(),
                   out_features_ptr,
                   tmp_out.data<float>(),
                   index,
                   phi::vectorize<int64_t>(out->values().dims()),
                   0,
                   false);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter");
  *(out->mutable_values()) = phi::Cast<float, phi::XPUContext>(
      dev_ctx, tmp_out, phi::DataType::FLOAT32);
}

template <typename T, typename Context>
void Conv3dCooKernel(const Context& dev_ctx,
                     const SparseCooTensor& x,
                     const DenseTensor& kernel,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const int groups,
                     const bool subm,
                     const std::string& key,
                     SparseCooTensor* out,
                     DenseTensor* rulebook,
                     DenseTensor* counter) {
  PD_VISIT_BASE_INTEGRAL_TYPES(x.indices().dtype(), "Conv3dCooCPUKernel", ([&] {
                                 Conv3dCooXPUKernel<T, data_t>(dev_ctx,
                                                               x,
                                                               kernel,
                                                               paddings,
                                                               dilations,
                                                               strides,
                                                               groups,
                                                               subm,
                                                               key,
                                                               out,
                                                               rulebook,
                                                               counter);
                               }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(
    conv3d_coo, XPU, ALL_LAYOUT, phi::sparse::Conv3dCooKernel, float) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
  kernel->OutputAt(1).SetDataType(paddle::DataType::INT32);
  kernel->OutputAt(2).SetDataType(paddle::DataType::INT32);
}
