// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law || agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES || CONDITIONS OF ANY KIND, either express || implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/api/lib/op_debug.h"
#include <string>
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace experimental {

void OpOutputDebugger::PrintOutput(const phi::DenseTensor *dt,
                                   Backend backend) {
  std::cout << "  Tensor: ";
  if (!dt || !dt->initialized()) {
    std::cout << "output is NOT INITIALIZED" << std::endl;
    return;
  }
  if (backend == Backend::CPU) {
    if (dt->dtype() == DataType::FLOAT32) {
      const float *cpu_res =
          static_cast<const float *>(dt->dy_acc_debug_data());
      double sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT32 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
    } else if (dt->dtype() == DataType::INT32) {
      const int32_t *cpu_res =
          static_cast<const int32_t *>(dt->dy_acc_debug_data());
      int64_t sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
    } else if (dt->dtype() == DataType::FLOAT64) {
      const double *cpu_res =
          static_cast<const double *>(dt->dy_acc_debug_data());
      double sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT64 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
    } else if (dt->dtype() == DataType::INT64) {
      const int64_t *cpu_res =
          static_cast<const int64_t *>(dt->dy_acc_debug_data());
      int64_t sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT64 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
    } else if (dt->dtype() == DataType::BOOL) {
      const bool *cpu_res = static_cast<const bool *>(dt->dy_acc_debug_data());
      std::string cpu_res_str;
      std::cout << "output BOOL " << std::endl;
      for (int i = 0; i < dt->numel(); i++) {
        if (cpu_res[i] == 0) {
          cpu_res_str = "false, ";
        } else if (cpu_res[i] > 0) {
          cpu_res_str = "true, ";
        } else {
          cpu_res_str = "ILLEGAL, ";
        }
        std::cout << cpu_res_str;
      }
      std::cout << "dimension is " << dt->dims() << "), place is "
                << dt->place() << std::endl;
    } else {
      std::cout << "output dtype " << dt->dtype() << " is NOT SUPPORTED"
                << std::endl;
    }
  } else if (backend == Backend::XPU) {
    if (dt->dtype() == DataType::FLOAT32) {
      float *cpu_res = new float[dt->numel()];
      xpu_memcpy(cpu_res,
                 dt->dy_acc_debug_data(),
                 sizeof(float) * dt->numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      double sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT32 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
      free(cpu_res);
    } else if (dt->dtype() == DataType::INT32) {
      int32_t *cpu_res = new int32_t[dt->numel()];
      xpu_memcpy(cpu_res,
                 dt->dy_acc_debug_data(),
                 sizeof(int32_t) * dt->numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      int64_t sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
      free(cpu_res);
    } else if (dt->dtype() == DataType::FLOAT64) {
      double *cpu_res = new double[dt->numel()];
      xpu_memcpy(cpu_res,
                 dt->dy_acc_debug_data(),
                 sizeof(double) * dt->numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      double sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT64 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
      free(cpu_res);
    } else if (dt->dtype() == DataType::INT64) {
      int64_t *cpu_res = new int64_t[dt->numel()];
      xpu_memcpy(cpu_res,
                 dt->dy_acc_debug_data(),
                 sizeof(int64_t) * dt->numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      int64_t sum_res = 0;
      for (int i = 0; i < dt->numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT64 " << sum_res << ", dimension is ("
                << dt->dims() << "), place is " << dt->place() << std::endl;
      free(cpu_res);
    } else if (dt->dtype() == DataType::BOOL) {
      uint8_t *cpu_res = new uint8_t[dt->numel()];
      xpu_memcpy(cpu_res,
                 dt->dy_acc_debug_data(),
                 sizeof(uint8_t) * dt->numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      std::string cpu_res_str;
      std::cout << "output BOOL " << std::endl;
      for (int i = 0; i < dt->numel(); i++) {
        if (cpu_res[i] == 0) {
          cpu_res_str = "false, ";
        } else if (cpu_res[i] > 0) {
          cpu_res_str = "true, ";
        } else {
          cpu_res_str = "ILLEGAL, ";
        }
        std::cout << cpu_res_str;
      }
      std::cout << "dimension is " << dt->dims() << "), place is "
                << dt->place() << std::endl;
      free(cpu_res);
    } else {
      std::cout << "output dtype " << dt->dtype() << " is NOT SUPPORTED"
                << std::endl;
    }
  } else {
    std::cout << "This tool does not support backends other than CPU and XPU!"
              << std::endl;
  }
}

void OpOutputDebugger::PrintOutput(const std::vector<phi::DenseTensor *> &v_t,
                                   Backend backend) {
  std::cout << "  std::vector<Tensor>:";
  if (v_t.size() == 0) {
    std::cout << " output is NOT INITIALIZED" << std::endl;
    return;
  }
  std::cout << std::endl;
  for (uint i = 0; i < (unsigned)v_t.size(); i++) {
    std::cout << "  ";
    PrintOutput(v_t[i], backend);
  }
}

}  // namespace experimental
}  // namespace paddle
