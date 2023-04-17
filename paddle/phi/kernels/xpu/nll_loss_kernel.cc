// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/nll_loss_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& labels,
                      const paddle::optional<DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      DenseTensor* out,
                      DenseTensor* total_weight) {
  VLOG(10) << "P3D xpu nll_loss_kernel.cc started kernel.";
  auto x_data = x.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  
  // label_data int64_t -> int32_t
  auto label_data = labels.data<int64_t>();
  int label_num = labels.numel();
  DenseTensor labels_int32;
  auto label_data_int32 = dev_ctx.template Alloc<int32_t>(&labels_int32, label_num * sizeof(int32_t));
  xpu::cast<int64_t, int32_t>(dev_ctx.x_context(), label_data, label_data_int32, label_num);


  //auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<float>() : nullptr;
  auto weight_data = nullptr;
  
  auto total_weight_data = dev_ctx.template Alloc<T>(total_weight);
  VLOG(10) << "P3D before total_weight_data assignment.";
  //*total_weight_data = 1;
  VLOG(10) << "P3D after total_weight_data assignment.";

  auto x_dim = x.dims();
  auto x_rank = x_dim.size();
  auto x_dims = x.dims();
  std::vector<int64_t> x_shape;
  VLOG(10) << "P3D before x_shape for loop.";
  for (int i = 0; i < x_rank; i++) {
    x_shape.push_back(x_dims[i]);
  }

  int reduction_id = 0;
  if (reduction == "none") {
    reduction_id = 0;
  }
  else if (reduction == "mean") {
    reduction_id = 1;
  }
  else if (reduction == "sum") {
    reduction_id = 2;
  }

  /*
  const auto batch_size = x_dims[0];
  const auto n_classes = x_dims[1];
  */

  VLOG(10) << "P3D before xpu::nll_loss.";
  int r = xpu::nll_loss(dev_ctx.x_context(),
            x_data,
            out_data,
            total_weight_data,
            x_shape,
            label_data_int32,
            weight_data,
            reduction_id,
            static_cast<int32_t>(ignore_index));
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "nll_loss");

  VLOG(10) << "P3D xpu nll_loss_kernel.cc finished kernel.";
}

template <typename T, typename Context>
void NllLossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   const paddle::optional<DenseTensor>& weight,
                   int64_t ignore_index,
                   const std::string& reduction,
                   DenseTensor* out) {
  DenseTensor total_weight;
  total_weight.set_meta(
      DenseTensorMeta(phi::CppTypeToDataType<T>::Type(), {1}));
  dev_ctx.template Alloc<T>(total_weight);
  VLOG(10) << "P3D nll_loss_kernel.cpp started nllloss kernel.";
  NllLossRawKernel(dev_ctx,
                   input,
                   label,
                   weight,
                   ignore_index,
                   reduction,
                   out,
                   &total_weight);
  VLOG(10) << "P3D nll_loss_kernel.cpp finished nllloss kernel.";
}
}  // namespace phi

// TODO(xiongkun): add the non-raw kernel register here.
PD_REGISTER_KERNEL(
    nll_loss, XPU, ALL_LAYOUT, phi::NllLossRawKernel, float) {}