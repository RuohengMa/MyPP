#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest

import numpy as np

sys.path.append("..")

from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle

paddle.enable_static()


def cubic_1(x, a):
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def value_bound(input, w, h, x, y):
    access_x = int(max(min(x, w - 1), 0))
    access_y = int(max(min(y, h - 1), 0))
    return input[:, :, access_y, access_x]


def bicubic_interp_np(
    input,
    out_h,
    out_w,
    out_size=None,
    align_corners=True,
    data_layout='NCHW',
):
    """trilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        if align_corners:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h

    if out_w > 1:
        if align_corners:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for k in range(out_h):
        if align_corners:
            h = ratio_h * k
        else:
            h = ratio_h * (k + 0.5) - 0.5
        input_y = np.floor(h)
        y_t = h - input_y
        for l in range(out_w):
            if align_corners:
                w = ratio_w * l
            else:
                w = ratio_w * (l + 0.5) - 0.5
            input_x = np.floor(w)
            x_t = w - input_x
            for i in range(batch_size):
                for j in range(channel):
                    coefficients = [0, 0, 0, 0]
                    for ii in range(4):
                        access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                        access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                        access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                        access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                        access_y = int(max(min(input_y - 1 + ii, in_h - 1), 0))

                        coefficients[ii] = cubic_interp1d(
                            input[i, j, access_y, access_x_0],
                            input[i, j, access_y, access_x_1],
                            input[i, j, access_y, access_x_2],
                            input[i, j, access_y, access_x_3],
                            x_t,
                        )
                    out[i, j, k, l] = cubic_interp1d(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                        y_t,
                    )
    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(input.dtype)


class XPUTestBicubicInterpV2Op(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'bicubic_interp_v2'
        self.use_dynamic_create_class = False

    class TestBicubicInterpOp(XPUOpTest):
        def setUp(self):
            self.out_size = None
            self.init_place()
            self.data_layout = 'NCHW'
            self.init_test_case()
            self.dtype = self.in_type
            self.op_type = "bicubic_interp_v2"
            input_np = np.random.random(self.input_shape).astype(self.dtype)

            if self.data_layout == "NCHW":
                in_h = self.input_shape[2]
                in_w = self.input_shape[3]
            else:
                in_h = self.input_shape[1]
                in_w = self.input_shape[2]
            scale_h = 0
            scale_w = 0

            if self.scale:
                if isinstance(self.scale, float) or isinstance(self.scale, int):
                    if self.scale > 0.0:
                        scale_h = scale_w = float(self.scale)
                if isinstance(self.scale, list) and len(self.scale) == 1:
                    scale_w = scale_h = self.scale[0]
                elif isinstance(self.scale, list) and len(self.scale) > 1:
                    scale_w = self.scale[1]
                    scale_h = self.scale[0]
                out_h = int(in_h * scale_h)
                out_w = int(in_w * scale_w)
            else:
                out_h = self.out_h
                out_w = self.out_w

            output_np = bicubic_interp_np(
                input_np,
                out_h,
                out_w,
                self.out_size,
                self.align_corners,
                self.data_layout,
            )
            self.inputs = {'X': input_np}
            if self.out_size is not None:
                self.inputs['OutSize'] = self.out_size

            self.attrs = {
                'out_h': self.out_h,
                'out_w': self.out_w,
                'interp_method': self.interp_method,
                'align_corners': self.align_corners,
                'align_mode': self.align_mode,
                'data_layout': self.data_layout,
            }
            if self.scale:
                if isinstance(self.scale, float) or isinstance(self.scale, int):
                    if self.scale > 0.0:
                        self.scale = [self.scale]
                if isinstance(self.scale, list) and len(self.scale) == 1:
                    self.scale = [self.scale[0], self.scale[0]]
                self.attrs['scale'] = self.scale
            self.outputs = {'Out': output_np}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        '''
        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True)
        '''

        def init_test_case(self):
            self.interp_method = 'bicubic'
            # self.input_shape = [2, 3, 5, 5]
            self.input_shape = [1, 1, 2, 2]
            self.out_h = 4
            self.out_w = 4
            self.scale = 0.0
            # self.out_size = np.array([3, 3]).astype("int32")
            self.align_corners = True
            self.align_mode = 1

        def init_place(self):
            self.place = paddle.XPUPlace(0)

    class TestBicubicInterpCase1(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [4, 1, 7, 8]
            self.out_h = 10
            self.out_w = 10
            self.scale = 0.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase2(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12
            self.scale = 0.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase3(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [1, 1, 32, 64]
            self.out_h = 64
            self.out_w = 32
            self.scale = 0.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase4(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [4, 1, 7, 8]
            self.out_h = 1
            self.out_w = 1
            self.scale = 0.0
            self.out_size = np.array([2, 2]).astype("int32")
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase5(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12
            self.scale = 0.0
            self.out_size = np.array([11, 11]).astype("int32")
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase6(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [1, 1, 32, 64]
            self.out_h = 64
            self.out_w = 32
            self.scale = 0.0
            self.out_size = np.array([65, 33]).astype("int32")
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpCase7(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [1, 1, 32, 64]
            self.out_h = 64
            self.out_w = 32
            self.scale = [2.0, 0.5]
            self.align_corners = False
            self.align_mode = 1

    class TestBicubicInterpSame(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 32, 64]
            self.out_h = 32
            self.out_w = 64
            self.scale = 0.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpActualShape(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.scale = 0.0
            self.out_size = np.array([66, 40]).astype("int32")
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpOtherMethod1(TestBicubicInterpOp):
        def set_align_mode(self):
            self.align_corners = False
            self.align_mode = 1

    class TestBicubicInterpWithMethod2(TestBicubicInterpOp):
        def set_align_mode(self):
            self.align_corners = False
            self.align_mode = 0

    class TestBicubicInterpWithMethod3(TestBicubicInterpOp):
        def set_align_mode(self):
            self.align_corners = True
            self.align_mode = 0

    class TestBicubicInterpScale1(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 7]
            self.out_h = 60
            self.out_w = 25
            self.scale = 2.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpScale2(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 7]
            self.out_h = 60
            self.out_w = 25
            self.scale = 1.0
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpScale3(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 7]
            self.out_h = 60
            self.out_w = 25
            self.scale = 1.5
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpScale4(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 7]
            self.out_h = 60
            self.out_w = 25
            self.scale = [1.5, 0.5]
            self.align_corners = True
            self.align_mode = 1

    class TestBicubicInterpZero(TestBicubicInterpOp):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 7]
            self.out_h = 60
            self.out_w = 25
            self.scale = 0.2
            self.align_corners = False
            self.align_mode = 0

    '''
    class TestBicubicInterpOp_attr_tensor(XPUOpTest):
        def setUp(self):
            self.out_size = None
            self.actual_shape = None
            self.init_test_case()
            self.init_place()
            self.op_type = "bilinear_interp_v2"
            self.dtype = self.in_type
            self.shape_by_1Dtensor = False
            self.scale_by_1Dtensor = False
            self.attrs = {
                'interp_method': self.interp_method,
                'align_corners': self.align_corners,
            }

            input_np = np.random.random(self.input_shape).astype(self.dtype)
            self.inputs = {'X': input_np}

            if self.scale_by_1Dtensor:
                self.inputs['Scale'] = np.array([self.scale]).astype("float32")
            elif self.scale:
                if isinstance(self.scale, float) or isinstance(self.scale, int):
                    if self.scale > 0:
                        scale_h = scale_w = float(self.scale)
                if isinstance(self.scale, list) and len(self.scale) == 1:
                    scale_w = scale_h = self.scale[0]
                elif isinstance(self.scale, list) and len(self.scale) > 1:
                    scale_w = self.scale[1]
                    scale_h = self.scale[0]
                out_h = int(self.input_shape[2] * scale_h)
                out_w = int(self.input_shape[3] * scale_w)
            else:
                out_h = self.out_h
                out_w = self.out_w

            if self.shape_by_1Dtensor:
                self.inputs['OutSize'] = self.out_size
            elif self.out_size is not None:
                size_tensor = []
                for index, ele in enumerate(self.out_size):
                    size_tensor.append(
                        ("x" + str(index), np.ones((1)).astype('int32') * ele)
                    )
                self.inputs['SizeTensor'] = size_tensor

            self.attrs['out_h'] = self.out_h
            self.attrs['out_w'] = self.out_w
            if self.scale:
                if isinstance(self.scale, float) or isinstance(self.scale, int):
                    if self.scale > 0:
                        self.scale = [self.scale]
                if isinstance(self.scale, list) and len(self.scale) == 1:
                    self.scale = [self.scale[0], self.scale[0]]
                self.attrs['scale'] = self.scale
            output_np = bilinear_interp_np(
                input_np,
                out_h,
                out_w,
                0,
                0,
                self.out_size,
                self.actual_shape,
                self.align_corners,
            )
            self.outputs = {'Out': output_np}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True)

        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [2, 3, 5, 5]
            self.out_h = 3
            self.out_w = 3
            self.scale = 0.0
            self.out_size = [3, 3]
            self.align_corners = True

        def init_place(self):
            self.place = paddle.XPUPlace(0)

    # out_size is a 1-D tensor
    class TestBicubicInterp_attr_tensor_Case1(
        TestBicubicInterpOp_attr_tensor
    ):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12
            self.scale = 0.0
            self.out_size = [8, 12]
            self.align_corners = True

    # scale is a 1-D tensor
    class TestBicubicInterp_attr_tensor_Case2(
        TestBicubicInterpOp_attr_tensor
    ):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.scale = 0.0
            self.out_size = np.array([66, 40]).astype("int32")
            self.align_corners = True
            self.shape_by_1Dtensor = True

    # scale is a 1-D tensor
    class TestBicubicInterp_attr_tensor_Case3(
        TestBicubicInterpOp_attr_tensor
    ):
        def init_test_case(self):
            self.interp_method = 'bicubic'
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.scale = 2.0
            self.out_size = None
            self.align_corners = True
            self.scale_by_1Dtensor = True

    '''


support_types = get_xpu_op_support_types('bicubic_interp_v2')
for stype in support_types:
    create_test_class(globals(), XPUTestBicubicInterpV2Op, stype)

if __name__ == "__main__":
    unittest.main()
