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

#include "paddle/phi/kernels/sparse/sparse_utils_kernel.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {
namespace sparse {

/*
template <typename ValueT, typename IndicesT>
__global__ void KernelCooToDense(const IndicesT* indices,
                                 const int64_t* sparse_offsets,
                                 const ValueT* data,
                                 ValueT* dense_data,
                                 const IndicesT non_zero_num,
                                 const int64_t base_offset,
                                 const int64_t sparse_dim) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < non_zero_num; i += gridDim.x * blockDim.x) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += indices[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      dense_data[index * base_offset + j] = data[i * base_offset + j];
    }
  }
}
*/

template <typename T, typename IntT>
void CooToDenseXPUKernel(const XPUContext& dev_ctx,
                         const SparseCooTensor& x,
                         DenseTensor* out) {
  const auto non_zero_num = x.nnz();
  const auto dense_dims = x.dims();
  const auto indices = x.indices();
  const auto values = x.values();
  const auto indices_dims = phi::vectorize<int>(indices.dims());
  int64_t sparse_dim = indices_dims[0];
  if (indices_dims.size() == 1) {
    sparse_dim = 1;
  }
  const int64_t dense_dim = x.dense_dim();

  // const T* x_data = values.data<T>();
  dev_ctx.template Alloc<T>(out);
  T* out_data = out->data<T>();
  xpu::constant(dev_ctx.x_context(), out_data, out->numel(), static_cast<T>(0));

  // temp solution
  // values
  DenseTensor tmp_x_values;
  phi::Copy(dev_ctx, x.values(), CPUPlace(), true, &tmp_x_values);
  const T* tmp_x_data = tmp_x_values.data<T>();
  // indices
  DenseTensor tmp_x_indices;
  phi::Copy(dev_ctx, x.indices(), CPUPlace(), true, &tmp_x_indices);
  const IntT* tmp_x_indices_data = tmp_x_indices.data<IntT>();
  // out
  DenseTensor tmp_out;
  phi::Copy(dev_ctx, *out, CPUPlace(), true, &tmp_out);
  T* tmp_out_data = tmp_out.data<T>();

  if (x.nnz() <= 0) {
    return;
  }

  int64_t base_offset = 1;
  for (int64_t i = 0; i < dense_dim; i++) {
    base_offset *= dense_dims[sparse_dim + i];
  }
  std::vector<int64_t> sparse_offsets(sparse_dim);
  int64_t offset = 1;
  for (int i = sparse_dim - 1; i >= 0; i--) {
    sparse_offsets[i] = offset;
    offset *= dense_dims[i];
  }

  /*
  KernelCooToDense<T, IntT>
      <<<8,
         64,
         dev_ctx.stream()>>>(indices.data<IntT>(),
                             sparse_offsets.data<int64_t>(),
                             x_data,
                             out_data,
                             non_zero_num,
                             base_offset,
                             sparse_dim);
  */

  for (auto i = 0; i < non_zero_num; i++) {
    int64_t index = 0;
    for (int j = 0; j < sparse_dim; j++) {
      index += tmp_x_indices_data[j * non_zero_num + i] * sparse_offsets[j];
    }

    for (int j = 0; j < base_offset; j++) {
      tmp_out_data[index * base_offset + j] = tmp_x_data[i * base_offset + j];
    }
  }

  // temp solution
  phi::Copy(dev_ctx, tmp_out, dev_ctx.GetPlace(), true, out);
}

template <typename T, typename Context>
void CooToDenseKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      DenseTensor* out) {
  PD_VISIT_BASE_INTEGRAL_TYPES(
      x.indices().dtype(), "CooToDenseXPUKernel", ([&] {
        CooToDenseXPUKernel<T, data_t>(dev_ctx, x, out);
      }));
}

}  // namespace sparse
}  // namespace phi

PD_REGISTER_KERNEL(coo_to_dense,
                   XPU,
                   ALL_LAYOUT,
                   phi::sparse::CooToDenseKernel,
                   float,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(sparse_coo_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::sparse::SparseCooTensorKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t) {}
