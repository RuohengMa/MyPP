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

#include "paddle/phi/kernels/sparse/batch_norm_kernel.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi {
namespace sparse {

/*
#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
template <typename T, typename Context>
void BatchNormCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& mean,
                        const DenseTensor& variance,
                        const DenseTensor& scale,
                        const DenseTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout,
                        bool use_global_stats,
                        bool trainable_statistics,
                        SparseCooTensor* y,
                        DenseTensor* mean_out,
                        DenseTensor* variance_out,
                        DenseTensor* saved_mean,
                        DenseTensor* saved_variance,
                        DenseTensor* reserve_space) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, y);
  phi::BatchNormKernel<T, Context>(dev_ctx,
                                  x.values(),
                                  mean,
                                  variance,
                                  scale,
                                  bias,
                                  is_test,
                                  momentum,
                                  epsilon,
                                  data_layout,
                                  use_global_stats,
                                  trainable_statistics,
                                  y->mutable_values(),
                                  mean_out,
                                  variance_out,
                                  saved_mean,
                                  saved_variance,
                                  reserve_space);

  DenseTensor x_values_cast =
    phi::Cast<T, Context>(dev_ctx, x.values(), phi::DataType::FLOAT32);
  DenseTensor y_values_cast =
    phi::Cast<T, Context>(dev_ctx, y->values(), phi::DataType::FLOAT32);

  DenseTensor mean_cast =
    phi::Cast<T, Context>(dev_ctx, mean, phi::DataType::FLOAT32);
  DenseTensor variance_cast =
    phi::Cast<T, Context>(dev_ctx, variance, phi::DataType::FLOAT32);
  DenseTensor scale_cast =
    phi::Cast<T, Context>(dev_ctx, scale, phi::DataType::FLOAT32);
  DenseTensor bias_cast =
    phi::Cast<T, Context>(dev_ctx, bias, phi::DataType::FLOAT32);

  DenseTensor mean_out_cast =
    phi::Cast<T, Context>(dev_ctx, *mean_out, phi::DataType::FLOAT32);
  DenseTensor variance_out_cast =
    phi::Cast<T, Context>(dev_ctx, *variance_out, phi::DataType::FLOAT32);
  DenseTensor saved_mean_cast;
  DenseTensor saved_variance_cast;
  DenseTensor reserve_space_cast;
  if (saved_mean->initialized()) {
    saved_mean_cast =
      phi::Cast<T, Context>(dev_ctx, *saved_mean, phi::DataType::FLOAT32);
  }
  else {
    saved_mean_cast = *saved_mean;
  }
  if (saved_variance->initialized()) {
    saved_variance_cast =
      phi::Cast<T, Context>(dev_ctx, *saved_variance, phi::DataType::FLOAT32);
  }
  else {
    saved_variance_cast = *saved_variance;
  }
  if (reserve_space->initialized()) {
    reserve_space_cast =
      phi::Cast<T, Context>(dev_ctx, *reserve_space, phi::DataType::FLOAT32);
  }
  else {
    reserve_space_cast = *reserve_space;
  }

  phi::BatchNormKernel<float, Context>(dev_ctx,
                                      x_values_cast,
                                      mean_cast,
                                      variance_cast,
                                      scale_cast,
                                      bias_cast,
                                      is_test,
                                      momentum,
                                      epsilon,
                                      data_layout,
                                      use_global_stats,
                                      trainable_statistics,
                                      &y_values_cast,
                                      &mean_out_cast,
                                      &variance_out_cast,
                                      &saved_mean_cast,
                                      &saved_variance_cast,
                                      &reserve_space_cast);

  if (x.dtype() == phi::DataType::FLOAT64) {
    *(y->mutable_values()) = phi::Cast<float, Context>(
      dev_ctx, y_values_cast, phi::DataType::FLOAT64);

    *mean_out = phi::Cast<float, Context>(
      dev_ctx, mean_out_cast, phi::DataType::FLOAT64);
    *variance_out = phi::Cast<float, Context>(
      dev_ctx, variance_out_cast, DataType::FLOAT64);
    *saved_mean = phi::Cast<float, Context>(
      dev_ctx, saved_mean_cast, phi::DataType::FLOAT64);
    *saved_variance = phi::Cast<float, Context>(
      dev_ctx, saved_variance_cast, phi::DataType::FLOAT64);
    if (reserve_space_cast.initialized()) {
      *reserve_space = phi::Cast<float, Context>(
        dev_ctx, reserve_space_cast, DataType::FLOAT64);
    }
  }


  y->SetIndicesDict(x.GetIndicesDict());
}
#else
*/
template <typename T, typename Context>
void BatchNormCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        const DenseTensor& mean,
                        const DenseTensor& variance,
                        const DenseTensor& scale,
                        const DenseTensor& bias,
                        bool is_test,
                        float momentum,
                        float epsilon,
                        const std::string& data_layout,
                        bool use_global_stats,
                        bool trainable_statistics,
                        SparseCooTensor* y,
                        DenseTensor* mean_out,
                        DenseTensor* variance_out,
                        DenseTensor* saved_mean,
                        DenseTensor* saved_variance,
                        DenseTensor* reserve_space) {
  EmptyLikeCooKernel<float, Context>(dev_ctx, x, y);
  phi::BatchNormKernel<float, Context>(dev_ctx,
                                       x.values(),
                                       mean,
                                       variance,
                                       scale,
                                       bias,
                                       is_test,
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       use_global_stats,
                                       trainable_statistics,
                                       y->mutable_values(),
                                       mean_out,
                                       variance_out,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space);
  y->SetIndicesDict(x.GetIndicesDict());
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(batch_norm_coo,
                   CPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   double) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

#if defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}
#endif

#if defined(PADDLE_WITH_CUDA)
PD_REGISTER_KERNEL(batch_norm_coo,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::BatchNormCooKernel,
                   float,
                   double,
                   phi::dtype::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
  }
}
#endif

#if defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
PD_REGISTER_KERNEL(
    batch_norm_coo, XPU, ALL_LAYOUT, phi::sparse::BatchNormCooKernel, float) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
#endif
