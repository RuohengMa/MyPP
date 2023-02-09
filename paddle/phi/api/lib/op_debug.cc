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
#include <string>
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
      const int32_t *cpu_res =
          static_cast<const int32_t *>(t.dy_acc_debug_data());
      int32_t sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      int32_t *cpu_res = new int32_t[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(int32_t) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      int32_t sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT32 " << sum_res << std::endl;
      free(cpu_res);
    } else {
      std::cout << "this tool does not support backends other than CPU and XPU"
                << std::endl;
    }
  } else if (t.dtype() == DataType::FLOAT64) {
    if (backend == Backend::CPU) {
      const double *cpu_res =
          static_cast<const double *>(t.dy_acc_debug_data());
      double sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT64 " << sum_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      double *cpu_res = new double[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(double) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      double sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output FLOAT64 " << sum_res << std::endl;
      free(cpu_res);
    } else {
      std::cout << "this tool does not support backends other than CPU and XPU"
                << std::endl;
    }
  } else if (t.dtype() == DataType::INT64) {
    if (backend == Backend::CPU) {
      const int64_t *cpu_res =
          static_cast<const int64_t *>(t.dy_acc_debug_data());
      int64_t sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT64 " << sum_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      int64_t *cpu_res = new int64_t[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(int64_t) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      int64_t sum_res = 0;
      for (int i = 0; i < t.numel(); i++) {
        sum_res += cpu_res[i];
      }
      std::cout << "output INT64 " << sum_res << std::endl;
      free(cpu_res);
    } else {
      std::cout << "this tool does not support backends other than CPU and XPU"
                << std::endl;
    }
  } else if (t.dtype() == DataType::BOOL) {
    if (backend == Backend::CPU) {
      const bool *cpu_res = static_cast<const bool *>(t.dy_acc_debug_data());
      std::string cpu_res_str;
      if (*cpu_res == 0) {
        cpu_res_str = "false";
      } else if (*cpu_res > 0) {
        cpu_res_str = "true";
      } else {
        std::cout << "output BOOL is ILLEGAL" << std::endl;
        return;
      }
      std::cout << "output BOOL " << *cpu_res << std::endl;
      return;
    } else if (backend == Backend::XPU) {
      uint8_t *cpu_res = new uint8_t[t.numel()];
      xpu_memcpy(cpu_res,
                 t.dy_acc_debug_data(),
                 sizeof(uint8_t) * t.numel(),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      std::string cpu_res_str;
      if (*cpu_res == 0) {
        cpu_res_str = "false";
      } else if (*cpu_res > 0) {
        cpu_res_str = "true";
      } else {
        std::cout << "output BOOL is ILLEGAL" << std::endl;
        free(cpu_res);
        return;
      }
      std::cout << "output BOOL " << cpu_res_str << std::endl;
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
        const int32_t *cpu_res =
            static_cast<const int32_t *>(t.dy_acc_debug_data());
        int32_t sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output INT32 " << sum_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        int32_t *cpu_res = new int32_t[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(int32_t) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        int32_t sum_res = 0;
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
  } else if (v_t[0].dtype() == DataType::FLOAT64) {
    for (int i = 0; (unsigned)i < v_t.size(); i++) {
      auto t = v_t[i];
      if (!t.initialized()) {
        std::cout << "output is NOT INITIALIZED" << std::endl;
        return;
      }
      if (backend == Backend::CPU) {
        const double *cpu_res =
            static_cast<const double *>(t.dy_acc_debug_data());
        double sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output FLOAT64 " << sum_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        double *cpu_res = new double[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(double) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        double sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output FLOAT64 " << sum_res << std::endl;
        free(cpu_res);
      } else {
        std::cout
            << "this tool does not support backends other than CPU and XPU"
            << std::endl;
      }
    }
  } else if (v_t[0].dtype() == DataType::INT64) {
    for (int i = 0; (unsigned)i < v_t.size(); i++) {
      auto t = v_t[i];
      if (!t.initialized()) {
        std::cout << "output is NOT INITIALIZED" << std::endl;
        return;
      }
      if (backend == Backend::CPU) {
        const int64_t *cpu_res =
            static_cast<const int64_t *>(t.dy_acc_debug_data());
        int64_t sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output INT64 " << sum_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        int64_t *cpu_res = new int64_t[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(int64_t) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        int64_t sum_res = 0;
        for (int i = 0; i < t.numel(); i++) {
          sum_res += cpu_res[i];
        }
        std::cout << "output INT64 " << sum_res << std::endl;
        free(cpu_res);
      } else {
        std::cout
            << "this tool does not support backends other than CPU and XPU"
            << std::endl;
      }
    }
  } else if (v_t[0].dtype() == DataType::BOOL) {
    for (int i = 0; (unsigned)i < v_t.size(); i++) {
      auto t = v_t[i];
      if (!t.initialized()) {
        std::cout << "output is NOT INITIALIZED" << std::endl;
        return;
      }
      if (backend == Backend::CPU) {
        const bool *cpu_res = static_cast<const bool *>(t.dy_acc_debug_data());
        std::string cpu_res_str;
        if (*cpu_res == 0) {
          cpu_res_str = "false";
        } else if (*cpu_res > 0) {
          cpu_res_str = "true";
        } else {
          std::cout << "output BOOL is ILLEGAL" << std::endl;
          return;
        }
        std::cout << "output BOOL " << *cpu_res << std::endl;
        return;
      } else if (backend == Backend::XPU) {
        uint8_t *cpu_res = new uint8_t[t.numel()];
        xpu_memcpy(cpu_res,
                   t.dy_acc_debug_data(),
                   sizeof(uint8_t) * t.numel(),
                   XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        std::string cpu_res_str;
        if (*cpu_res == 0) {
          cpu_res_str = "false";
        } else if (*cpu_res > 0) {
          cpu_res_str = "true";
        } else {
          std::cout << "output BOOL is ILLEGAL" << std::endl;
          free(cpu_res);
          return;
        }
        std::cout << "output BOOL " << cpu_res_str << std::endl;
        free(cpu_res);
      } else {
        std::cout
            << "this tool does not support backends other than CPU and XPU"
            << std::endl;
      }
    }
  } else {
    std::cout << "output dtype " << v_t[0].dtype() << " is NOT SUPPORTED"
              << std::endl;
  }
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
