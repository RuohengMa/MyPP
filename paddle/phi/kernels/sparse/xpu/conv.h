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

#pragma once

#include <set>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/sparse_coo_tensor.h"
#include "paddle/phi/core/tensor_meta.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/sparse/conv_kernel.h"

namespace phi {
namespace sparse {

using Dims4D = phi::funcs::sparse::Dims4D;

// such as: kernel(3, 3, 3), kernel_size = 27
// counter_per_weight: (kernel_size)
// TODO(zhangkaihuo): optimize performance with multithreading
template <typename T, typename Context, typename IntT = int>
void ProductRuleBook(const Context& dev_ctx,
                     const SparseCooTensor& x_xpu,
                     const std::vector<int>& kernel_sizes,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations,
                     const std::vector<int>& strides,
                     const DDim& out_dims,
                     const bool subm,
                     DenseTensor* rulebook_xpu,
                     int* counter_per_kernel) {
  // temporary solution till xpu implementation of ProductRuleBook is finished
  SparseCooTensor x;
  phi::Copy(dev_ctx, x_xpu, phi::CPUPlace(), true, &x);

  const int64_t non_zero_num = x.nnz();
  const auto& indices = x.indices();

  const IntT* indices_ptr = indices.data<IntT>();
  int kernel_size = kernel_sizes[0] * kernel_sizes[1] * kernel_sizes[2];
  memset(counter_per_kernel, 0, kernel_size * sizeof(int));

  int rulebook_len = 0;
  // calc the rulebook_len
  const auto& x_dims = x.dims();
  const Dims4D c_x_dims(x_dims[0], x_dims[3], x_dims[2], x_dims[1]);
  const Dims4D c_kernel_dims(
      1, kernel_sizes[2], kernel_sizes[1], kernel_sizes[0]);
  const Dims4D c_out_dims(out_dims[0], out_dims[3], out_dims[2], out_dims[1]);
  const Dims4D c_paddings(1, paddings[2], paddings[1], paddings[0]);
  const Dims4D c_strides(1, strides[2], strides[1], strides[0]);
  const Dims4D c_dilations(1, dilations[2], dilations[1], dilations[0]);

  std::set<IntT> hash_in;
  if (subm) {
    for (int i = 0; i < non_zero_num; i++) {
      IntT batch = indices_ptr[i];
      IntT in_z = indices_ptr[i + non_zero_num];
      IntT in_y = indices_ptr[i + 2 * non_zero_num];
      IntT in_x = indices_ptr[i + 3 * non_zero_num];
      IntT index = phi::funcs::sparse::PointToIndex<DDim>(
          batch, in_x, in_y, in_z, x_dims);
      hash_in.insert(index);
    }
  }

  auto f_calc_rulebook = [&](IntT* rulebook_ptr) {
    int kernel_index = 0, rulebook_index = 0;
    for (int kz = 0; kz < kernel_sizes[0]; kz++) {
      for (int ky = 0; ky < kernel_sizes[1]; ky++) {
        for (int kx = 0; kx < kernel_sizes[2]; kx++) {
          ++kernel_index;
          for (int64_t i = 0; i < non_zero_num; i++) {
            IntT batch = indices_ptr[i];
            IntT in_z = indices_ptr[i + non_zero_num];
            IntT in_y = indices_ptr[i + 2 * non_zero_num];
            IntT in_x = indices_ptr[i + 3 * non_zero_num];
            IntT out_z = (in_z + paddings[0] - kz * dilations[0]) / strides[0];
            IntT out_y = (in_y + paddings[1] - ky * dilations[1]) / strides[1];
            IntT out_x = (in_x + paddings[2] - kx * dilations[2]) / strides[2];
            if (phi::funcs::sparse::Check(c_x_dims,
                                          c_kernel_dims,
                                          c_paddings,
                                          c_dilations,
                                          c_strides,
                                          in_x,
                                          in_y,
                                          in_z,
                                          kx,
                                          ky,
                                          kz)) {
              if (subm) {
                IntT out_index = phi::funcs::sparse::PointToIndex<DDim>(
                    batch, out_x, out_y, out_z, out_dims);
                if (hash_in.find(out_index) == hash_in.end()) {
                  continue;
                }
              }

              if (rulebook_ptr == nullptr) {
                counter_per_kernel[kernel_index - 1] += 1;
                ++rulebook_len;
              } else {
                rulebook_ptr[rulebook_index] = kernel_index - 1;
                rulebook_ptr[rulebook_index + rulebook_len] = i;  // in_i
                rulebook_ptr[rulebook_index + rulebook_len * 2] =
                    phi::funcs::sparse::PointToIndex<DDim>(
                        batch, out_x, out_y, out_z, out_dims);  // out_index
                ++rulebook_index;
              }
            }
          }
        }
      }
    }
  };

  f_calc_rulebook(nullptr);
  // alloc the rulebook
  *rulebook_xpu =
      phi::Empty(dev_ctx,
                 DenseTensorMeta(phi::CppTypeToDataType<IntT>::Type(),
                                 {3, rulebook_len},
                                 DataLayout::NCHW));
  DenseTensor rulebook;
  phi::Copy(dev_ctx, *rulebook_xpu, phi::CPUPlace(), true, &rulebook);
  IntT* rulebook_ptr = rulebook.data<IntT>();
  f_calc_rulebook(rulebook_ptr);

  // temporary solution till xpu implementation of ProductRuleBook is finished
  phi::Copy(dev_ctx, rulebook, dev_ctx.GetPlace(), true, rulebook_xpu);
}

template <typename T, typename Context, typename IntT = int>
void UpdateRulebookAndOutIndex(const Context& dev_ctx,
                               const SparseCooTensor& x_xpu,
                               const int kernel_size UNUSED,
                               const int out_channels,
                               const DDim& out_dims,
                               DenseTensor* rulebook_xpu,
                               SparseCooTensor* out) {
  // temporary solution till xpu implementation of ProductRuleBook is finished
  SparseCooTensor x;
  phi::Copy(dev_ctx, x_xpu, phi::CPUPlace(), true, &x);
  DenseTensor rulebook;
  phi::Copy(dev_ctx, *rulebook_xpu, phi::CPUPlace(), true, &rulebook);

  std::set<IntT> out_indexs;
  int n = rulebook.dims()[1];
  IntT* rulebook_ptr = rulebook.data<IntT>();
  for (int i = 0; i < n; i++) {
    out_indexs.insert(rulebook_ptr[i + n * 2]);
  }

  int out_non_zero_num = out_indexs.size();
  const int64_t sparse_dim = 4;
  DenseTensorMeta indices_meta(phi::CppTypeToDataType<IntT>::Type(),
                               {sparse_dim, out_non_zero_num},
                               DataLayout::NCHW);
  DenseTensorMeta values_meta(
      x.dtype(), {out_non_zero_num, out_channels}, x.values().layout());
  phi::DenseTensor out_indices_xpu =
      phi::Empty(dev_ctx, std::move(indices_meta));
  phi::DenseTensor out_values_xpu = phi::Empty(dev_ctx, std::move(values_meta));
  phi::DenseTensor out_indices, out_values;
  phi::Copy(dev_ctx, out_indices_xpu, phi::CPUPlace(), true, &out_indices);
  phi::Copy(dev_ctx, out_values_xpu, phi::CPUPlace(), true, &out_values);
  IntT* out_indices_ptr = out_indices.data<IntT>();
  int i = 0;
  for (auto it = out_indexs.begin(); it != out_indexs.end(); it++, i++) {
    const IntT index = *it;
    IntT batch, x, y, z;
    phi::funcs::sparse::IndexToPoint<DDim>(index, out_dims, &batch, &x, &y, &z);
    out_indices_ptr[i] = batch;
    out_indices_ptr[i + out_non_zero_num] = z;
    out_indices_ptr[i + out_non_zero_num * 2] = y;
    out_indices_ptr[i + out_non_zero_num * 3] = x;
  }
  for (i = 0; i < n; i++) {
    IntT out_index = rulebook_ptr[i + n * 2];
    rulebook_ptr[i + n * 2] =
        std::distance(out_indexs.begin(), out_indexs.find(out_index));
  }

  // temporary solution till xpu implementation of ProductRuleBook is finished
  phi::Copy(dev_ctx, out_indices, dev_ctx.GetPlace(), true, &out_indices_xpu);
  phi::Copy(dev_ctx, out_values, dev_ctx.GetPlace(), true, &out_values_xpu);

  out->SetMember(out_indices_xpu, out_values_xpu, out_dims, true);
}

}  // namespace sparse
}  // namespace phi
