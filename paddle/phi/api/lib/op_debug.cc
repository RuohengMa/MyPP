// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/api/lib/op_debug.h"
#include "paddle/fluid/platform/device/device_wrapper.h"

namespace paddle {
namespace experimental {

void OpOutputDebugger::PrintOutput(const Tensor &t,
                                   Backend backend,
                                   bool optional) {
  if (optional) {
    std::cout << "  paddle::optional<Tensor>: ";
  } else {
    std::cout << "  Tensor: ";
  }
  if (!t.initialized()) {
    std::cout << "output is NOT INITIALIZED" << std::endl;
    return;
  }
  if (t.dtype() == DataType::FLOAT32) {
    if (backend == Backend::CPU) {
      const float *cpu_res = static_cast<const float *>(t.dy_acc_debug_data());
      float sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT32 " << sum_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      float *cpu_res = new float[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(float) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      float sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT32 " << sum_res << std::endl;
      free(cpu_res);
    } else {
      std::cout << "this tool does not support backends other than CPU and XPU"
                << std::endl;
    }
  } else if (t.dtype() == DataType::INT32) {
    if (backend == Backend::CPU) {
      const int *cpu_res = static_cast<const int *>(t.dy_acc_debug_data());
      int sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      int *cpu_res = new int[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(int) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      int sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << std::endl;
      free(cpu_res);
    } else {
      std::cout << "this tool does not support backends other than CPU and XPU"
                << std::endl;
    }
  } else {
    std::cout << "output dtype " << t.dtype() << " is NOT SUPPORTED"
              << std::endl;
  }
}
void OpOutputDebugger::PrintOutput(const std::vector<Tensor> &v_t,
                                   Backend backend,
                                   bool optional) {
  if (optional) {
    std::cout << "  paddle::optional<std::vector<Tensor>>: ";
  } else {
    std::cout << "  std::vector<Tensor>: ";
  }
  if (v_t[0].dtype() == DataType::FLOAT32) {
    for (int i = 0; (unsigned)i < v_t.size(); i++) {
      auto t = v_t[i];
      if (!t.initialized()) {
        std::cout << "output is NOT INITIALIZED" << std::endl;
        return;
      }
      if (backend == Backend::CPU) {
        const float *cpu_res =
            static_cast<const float *>(t.dy_acc_debug_data());
        float sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output FLOAT32 " << sum_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        float *cpu_res = new float[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(float) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        float sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output FLOAT32 " << sum_res << std::endl;
        free(cpu_res);
      } else {
        std::cout
            << "this tool does not support backends other than CPU and XPU"
            << std::endl;
      }
    }
  } else if (v_t[0].dtype() == DataType::INT32) {
    for (int i = 0; (unsigned)i < v_t.size(); i++) {
      auto t = v_t[i];
      if (!t.initialized()) {
        std::cout << "output is NOT INITIALIZED" << std::endl;
        return;
      }
      if (backend == Backend::CPU) {
        const int *cpu_res = static_cast<const int *>(t.dy_acc_debug_data());
        int sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output INT32 " << sum_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        int *cpu_res = new int[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(int) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        int sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output INT32 " << sum_res << std::endl;
        free(cpu_res);
      } else {
        std::cout
            << "this tool does not support backends other than CPU and XPU"
            << std::endl;
      }
    }
  } else {
    std::cout << "output dtype " << v_t[0].dtype() << " is NOT SUPPORTED";
  }
  std::cout << std::endl;
}
void OpOutputDebugger::PrintOutput(const paddle::optional<Tensor> &t,
                                   Backend backend) {
  if (t) {
    OpOutputDebugger::PrintOutput(*t, backend, true);
  } else {
    std::cout << "  paddle::optional<Tensor>: NOT INITIALIZED" << std::endl;
  }
}
void OpOutputDebugger::PrintOutput(
    const paddle::optional<std::vector<Tensor>> &v_t, Backend backend) {
  if (v_t) {
    OpOutputDebugger::PrintOutput(*v_t, backend, true);
  } else {
    std::cout << "  paddle::optional<std::vector<Tensor>>: NOT INITIALIZED"
              << std::endl;
  }
}
}  // namespace experimental
}  // namespace paddle
