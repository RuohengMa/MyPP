# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import re

import yaml
from api_base import PREFIX_TENSOR_NAME, BaseAPI, set_prefix_tensor_name

PREFIX_OUTPUT = "debug_"

# skip dropout_grad since it throws ValueError due to uint8 -> float32 conversion
SKIP_LIST = ["dropout_grad"]


class BackwardAPI(BaseAPI):
    def __init__(self, backward_item_yaml):
        super().__init__(backward_item_yaml)
        self.check_args(backward_item_yaml['forward'])
        self.no_need_buffer = self.parse_no_need_buffer(backward_item_yaml)

    def get_api_name(self, api_item_yaml):
        return api_item_yaml['backward_op']

    def parse_forward_config(self, forward_config):
        # api_name (const Tensor& input, ... , int attr, ...) -> Tensor(out)
        result = re.search(
            r"(?P<op>[a-z][a-z0-9_]+)\s*(?P<args>\([^\)]+\))\s*->\s*(?P<outputs>.+)",
            forward_config,
        )
        api = result.group('op')
        (
            _,
            outputs,
            _,
        ) = self.parse_output(self.api, result.group('outputs'))
        outputs = [item.split('@')[0] for item in outputs]
        fw_inputs, fw_attrs = self.parse_input_and_attr(
            api, result.group('args')
        )

        return api, fw_inputs, fw_attrs, outputs

    def parse_no_need_buffer(self, api_item_yaml):
        no_need_buffer = []
        if 'no_need_buffer' in api_item_yaml:
            no_need_buffer = [
                item.strip()
                for item in api_item_yaml['no_need_buffer'].split(',')
            ]
        return no_need_buffer

    def check_args(self, forward_config):
        # parse the forward and backward config
        _, fw_inputs, fw_attrs, fw_outputs = self.parse_forward_config(
            forward_config
        )

        # check the inputs of backward
        for input in self.inputs['names']:
            if input not in fw_inputs['names'] and input not in fw_outputs:
                if input.endswith('_grad'):
                    original_name = input[:-5]
                    assert (
                        original_name in fw_outputs
                    ), f"{self.api} : Input Tensor error: the input tensor({input}) of backward should be an input or output or grad of output in forward api. \
                         Please check the forward of {self.api} in yaml."

        # check the attributes of backward
        for attr in self.attrs['names']:
            assert (
                attr in fw_attrs['names']
                and self.attrs['attr_info'][attr][0]
                == fw_attrs['attr_info'][attr][0]
            ) or self.attrs['attr_info'][attr][
                1
            ] is not None, f"{self.api} : Attribute error: The attribute({attr}) of backward isn't consistent with forward api or doesn't have default value. \
                 Please check the args of {self.api} in yaml."

        # check the output of backward
        assert len(self.outputs['types']) <= len(
            fw_inputs['names']
        ), f"{self.api} : Output error: The number of outputs should be less then the number of inputs of forward api. \
             Please check the output of {self.api} in yaml."

    def get_declare_args(self, inplace_flag=False):
        return self.get_define_args()

    def get_define_args(self, inplace_flag=False):
        out_type_map = {
            'Tensor': 'Tensor*',
            'std::vector<Tensor>': 'std::vector<Tensor*>',
        }
        intputs_and_attrs = super().get_define_args()
        outs = []
        for i, name in enumerate(self.outputs['names']):
            outs.append(
                out_type_map[self.outputs['types'][i]]
                + ' '
                + name.split('@')[0]
            )
        result = intputs_and_attrs + ', ' + ", ".join(outs)
        return result

    def gene_return_code(self):
        return ""

    def print_op_input(self, code_indent=''):
        size = len(self.inputs['names'])
        kernel_name = self.XPU_DY_ACC_DEBUG_kernel_name
        code_indent = self.XPU_DY_ACC_DEBUG_code_indent
        debug_code = f"""
{code_indent}  if (std::getenv("XPU_DY_ACC_DEBUG") != nullptr || std::getenv("XPU_DY_ACC_DEBUG_INPUT") != nullptr) {{
{code_indent}    auto dy_acc_debug_dev_place = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
{code_indent}    if (platform::is_xpu_place(phi::TransToPhiPlace(dy_acc_debug_dev_place))) {{
{code_indent}      dev_ctx->Wait();
{code_indent}    }}
{code_indent}    std::string inplace_string = inplace ? "true" : "false";
{code_indent}    std::cout << std::endl;
{code_indent}    std::cout << "op_name: " << phi::TransToFluidOpName("{kernel_name}") << ", global_id: b-" << global_id << ", place: " << dy_acc_debug_dev_place << ", dtype: " << kernel_data_type << ", inplace: " << inplace_string << std::endl;
{code_indent}    global_id += 1;
{code_indent}    std::cout << "input: " << std::endl;"""
        for i in range(size):
            input_name = self.inputs['names'][i]
            if (
                self.kernel['param'] is None
                and input_name in (self.inputs['names'] + self.attrs['names'])
                or input_name in self.kernel['param']
            ):
                if (
                    self.inputs['input_info'][input_name]
                    == "const std::vector<Tensor>&"
                    or self.inputs['input_info'][input_name]
                    == "const paddle::optional<std::vector<Tensor>>&"
                ) and input_name not in self.inplace_map.values():
                    debug_code += f"""
{code_indent}  OpOutputDebugger::PrintOutput({PREFIX_TENSOR_NAME}{self.inputs['names'][i]}_vec, dy_acc_debug_dev_place);"""
                else:
                    debug_code += f"""
{code_indent}  OpOutputDebugger::PrintOutput({PREFIX_TENSOR_NAME}{self.inputs['names'][i]}, dy_acc_debug_dev_place);"""
        debug_code += f"""
{code_indent}  }}
{code_indent}"""
        return debug_code

    def print_op_output(self):
        size = len(self.kernel_outputs)
        kernel_name = self.XPU_DY_ACC_DEBUG_kernel_name
        code_indent = self.XPU_DY_ACC_DEBUG_code_indent
        debug_code = f"""
{code_indent}  if (std::getenv("XPU_DY_ACC_DEBUG_OUTPUT") != nullptr) {{
{code_indent}    auto dy_acc_debug_dev_place = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
{code_indent}    if (platform::is_xpu_place(phi::TransToPhiPlace(dy_acc_debug_dev_place))) {{
{code_indent}      dev_ctx->Wait();
{code_indent}    }}
{code_indent}    std::string inplace_string = inplace ? "true" : "false";
{code_indent}    std::cout << std::endl;
{code_indent}    std::cout << "op_name: " << phi::TransToFluidOpName("{kernel_name}") << ", global_id: b-" << global_id << ", place: " << dy_acc_debug_dev_place << ", dtype: " << kernel_data_type << ", inplace: " << inplace_string << std::endl;
{code_indent}    global_id += 1;
{code_indent}  }}
{code_indent}  if (std::getenv("XPU_DY_ACC_DEBUG") != nullptr || std::getenv("XPU_DY_ACC_DEBUG_OUTPUT") != nullptr) {{
{code_indent}    auto dy_acc_debug_dev_place = kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend;
{code_indent}    std::cout << "CPU output: " << std::endl;"""
        if self.api not in SKIP_LIST:
            for i in range(size):
                debug_code += f"""
{code_indent}    OpOutputDebugger::PrintOutput({self.kernel_debug_outputs[i]}, Backend::CPU);"""
        else:
            debug_code += f"""
{code_indent}    std::cout << "    api in skip list, skipped." << std::endl;"""
        debug_code += f"""
{code_indent}    std::cout << "XPU output: " << std::endl;"""
        for i in range(size):
            debug_code += f"""
{code_indent}    OpOutputDebugger::PrintOutput({self.kernel_outputs[i]}, dy_acc_debug_dev_place);"""
        debug_code += f"""
{code_indent}  }}
{code_indent}"""
        return debug_code

    def gene_debug_input(self, kernel_dispatch, code_indent):
        if self.api in SKIP_LIST:
            return ""

        set_prefix_tensor_name("debug_input_")
        global PREFIX_TENSOR_NAME
        PREFIX_TENSOR_NAME = "debug_input_"

        # reuse code and modify it using string substitution
        input_tensors, kernel_args, kernel_signature = self.get_kernel_args(
            kernel_dispatch, code_indent
        )

        # prefix input name with 'debug_'
        input_tensors = re.sub(
            r"kernel\.InputAt\(([0-9]+)\)",
            r"{Backend::CPU, kernel.InputAt(\1).layout, kernel.InputAt(\1).dtype, kernel.InputAt(\1).type_index}",
            input_tensors,
        )

        # delete code for op info recording since it is not necessary when debugging accuracy issues
        input_tensors = re.sub(
            r"if\(phi::RecordOpInfoSupplement::IsEnabled\(\)\)((.|\n)*)",
            "",
            input_tensors,
        )

        # all debug_input should be a deep copy of the api input tensor to avoid divergence of CPU and XPU inputs
        # PrepareData and PrepareDataForSelectedRows have already fulfilled this requirement, while TensorToConstDenseTensorPtr does not
        # so it is necessary to modify it
        pattern = rf"std::vector\<const phi::DenseTensor\*\> {PREFIX_TENSOR_NAME}([a-z|_|0-9]+) = TensorToConstDenseTensorPtr\([a-z|_|0-9]+\);"
        sub_str = rf"""
{code_indent}  auto {PREFIX_TENSOR_NAME}\1_vec = PrepareData(\1, {{Backend::CPU, DataLayout::UNDEFINED, DataType::UNDEFINED, typeid(int)}}, {{false, false, true, false}});
{code_indent}  std::vector<const phi::DenseTensor*> {PREFIX_TENSOR_NAME}\1({PREFIX_TENSOR_NAME}\1_vec->size());
{code_indent}  for (size_t i = 0; i < {PREFIX_TENSOR_NAME}\1.size(); ++i) {{
{code_indent}    {PREFIX_TENSOR_NAME}\1[i] = &{PREFIX_TENSOR_NAME}\1_vec->at(i);
{code_indent}  }}"""
        input_tensors = re.sub(pattern, sub_str, input_tensors)

        pattern = rf"paddle::optional\<std::vector\<const phi::DenseTensor\*\>\> {PREFIX_TENSOR_NAME}([a-z|_|0-9]+) = TensorToConstDenseTensorPtr\([a-z|_|0-9]+\);"
        sub_str = rf"""
{code_indent}  auto {PREFIX_TENSOR_NAME}\1_vec = PrepareData(\1, {{Backend::CPU, DataLayout::UNDEFINED, DataType::UNDEFINED, typeid(int)}}, {{false, false, true, false}});
{code_indent}  paddle::optional<std::vector<const phi::DenseTensor*>> {PREFIX_TENSOR_NAME}\1;
{code_indent}  if ({PREFIX_TENSOR_NAME}\1_vec){{
{code_indent}    {PREFIX_TENSOR_NAME}\1 = paddle::optional<std::vector<const phi::DenseTensor*>>({PREFIX_TENSOR_NAME}\1_vec->size());
{code_indent}    for (size_t i = 0; i < {PREFIX_TENSOR_NAME}\1_vec->size(); ++i) {{
{code_indent}      {PREFIX_TENSOR_NAME}\1->at(i) = &{PREFIX_TENSOR_NAME}\1_vec->at(i);
{code_indent}    }}
{code_indent}  }}"""

        input_tensors = re.sub(pattern, sub_str, input_tensors)

        set_prefix_tensor_name("input_")
        PREFIX_TENSOR_NAME = "input_"

        return input_tensors

    def gene_debug_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        if self.api in SKIP_LIST:
            return ""

        kernel_output = []
        output_names = []
        output_create = ""

        if len(out_dtype_list) == 1:
            kernel_output.append(f'{PREFIX_OUTPUT}kernel_out')
            output_names.append(f'{PREFIX_OUTPUT}kernel_out')

            if out_dtype_list[0] == 'std::vector<Tensor>':
                assert (
                    self.outputs['out_size_expr'] is not None
                ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto {PREFIX_OUTPUT}kernel_out = std::vector<phi::DenseTensor*>({self.outputs['names'][0]}.size(), std::make_shared<phi::DenseTensor>().get());"""
                )

            else:
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto {PREFIX_OUTPUT}kernel_out = new phi::DenseTensor();"""
                )

        elif len(out_dtype_list) > 1:
            output_create = ""
            for i, out_type_item in enumerate(out_dtype_list):
                kernel_output.append(f'{PREFIX_OUTPUT}kernel_out_{i}')
                output_names.append(f'{PREFIX_OUTPUT}kernel_out_{i}')
                set_out_func = (
                    'SetKernelOutput'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'SetSelectedRowsKernelOutput'
                )
                if out_type_item == 'Tensor':
                    '''
                                        if (
                                            inplace_flag
                                            and self.inplace_map is not None
                                            and self.outputs['names'][i] in self.inplace_map
                                        ):
                                            output_create = (
                                                output_create
                                                + f"""
                    {code_indent}  auto {PREFIX_OUTPUT}{self.outputs['names'][i]} = {PREFIX_OUTPUT}{self.inplace_map[self.outputs['names'][i]]};
                    {code_indent}  *{PREFIX_OUTPUT}{self.outputs['names'][i]} = {PREFIX_OUTPUT}{self.inplace_map[self.outputs['names'][i]]};"""
                                            )
                    '''
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto {PREFIX_OUTPUT}kernel_out_{i} = new phi::DenseTensor();"""
                    )

                else:
                    '''
                                        if (
                                            inplace_flag
                                            and self.inplace_map is not None
                                            and self.outputs['names'][i] in self.inplace_map
                                        ):
                                            output_create = (
                                                output_create
                                                + f"""
                    {code_indent}  *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""
                                            )
                    '''
                    assert (
                        self.outputs['out_size_expr'][i] is not None
                    ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto {PREFIX_OUTPUT}kernel_out_{i} = std::vector<phi::DenseTensor*>({self.outputs['names'][i]}.size(), std::make_shared<phi::DenseTensor>().get());"""
                    )

        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )

        self.kernel_debug_outputs = kernel_output

        return output_create

    def gene_debug_infer_meta(self, kernel_output_names, code_indent) -> str:
        if self.api in SKIP_LIST:
            return ""

        PREFIX_OUTPUT = "debug_"
        PREFIX_META_TENSOR_NAME = "meta_"

        input_names = self.inputs['names']
        attr_names = self.attrs['names']
        infer_meta = self.infer_meta

        infer_meta_params = (
            infer_meta['param']
            if infer_meta['param'] is not None
            else input_names + attr_names
        )
        # generate meta tensors
        meta_tensor_code = ""
        param_code = ""
        for param in infer_meta_params:
            if param in input_names:
                if self.inputs['input_info'][param] == "const Tensor&":
                    param_code = (
                        param_code
                        + "MakeMetaTensor(*"
                        + PREFIX_OUTPUT
                        + PREFIX_TENSOR_NAME
                        + param
                        + "), "
                    )
                elif (
                    self.inputs['input_info'][param]
                    == "const std::vector<Tensor>&"
                ):
                    meta_tensor_code = (
                        meta_tensor_code
                        + f"""
{code_indent}  auto {PREFIX_OUTPUT}{param}_meta_vec = MakeMetaTensor({PREFIX_OUTPUT}{PREFIX_TENSOR_NAME}{param});
{code_indent}  std::vector<const phi::MetaTensor*> {PREFIX_OUTPUT}{param}_metas({PREFIX_OUTPUT}{param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {PREFIX_OUTPUT}{param}_meta_vec.size(); ++i) {{
{code_indent}    {PREFIX_OUTPUT}{param}_metas[i] = &{PREFIX_OUTPUT}{param}_meta_vec[i];
{code_indent}  }}
"""
                    )
                    param_code = param_code + PREFIX_OUTPUT + param + "_metas, "
                elif (
                    self.inputs['input_info'][param]
                    == "const paddle::optional<std::vector<Tensor>>&"
                ):
                    meta_tensor_code = (
                        meta_tensor_code
                        + f"""
{code_indent}  auto {PREFIX_OUTPUT}{param}_meta_vec = MakeMetaTensor({PREFIX_OUTPUT}{PREFIX_TENSOR_NAME}{param});
{code_indent}  paddle::optional<std::vector<const phi::MetaTensor*>> {PREFIX_OUTPUT}{param}_metas({PREFIX_OUTPUT}{param}_meta_vec.size());
{code_indent}  for (size_t i = 0; i < {PREFIX_OUTPUT}{param}_meta_vec.size(); ++i) {{
{code_indent}    {PREFIX_OUTPUT}{param}_metas->at(i) = &{PREFIX_OUTPUT}{param}_meta_vec[i];
{code_indent}  }}
"""
                    )
                    param_code = param_code + PREFIX_OUTPUT + param + "_metas, "
                elif param in self.optional_vars:
                    param_code = (
                        param_code
                        + "MakeMetaTensor("
                        + PREFIX_OUTPUT
                        + PREFIX_TENSOR_NAME
                        + param
                        + "), "
                    )
                else:
                    raise ValueError(
                        f"{self.api} : Param of infer_meta error : {self.inputs['input_info'][param]} type is not supported."
                    )
            elif param in attr_names:
                param_code = param_code + param + ", "
            elif isinstance(param, str):
                param_code = param_code + "\"" + param + "\", "
            elif isinstance(param, bool):
                param_code = param_code + str(param).lower() + ", "
            else:
                param_code = param_code + str(param) + ", "

        for i, out_name in enumerate(kernel_output_names):
            if self.outputs['types'][i] == 'std::vector<Tensor>':
                '''
                if self.outputs['names'][i] in self.inplace_map and self.inplace_map[self.outputs['names'][i]] in self.optional_vars:
                    out_name = re.sub("debug_", "", out_name)
                    param_code = param_code + out_name + '_metas, '
                '''
                if True:
                    meta_tensor_code = (
                        meta_tensor_code
                        + f"""
{code_indent}  auto {PREFIX_OUTPUT}{out_name}_{PREFIX_META_TENSOR_NAME}vec = MakeMetaTensor({PREFIX_OUTPUT}{out_name});
{code_indent}  std::vector<phi::MetaTensor*> {PREFIX_OUTPUT}{out_name}_metas({PREFIX_OUTPUT}{out_name}_{PREFIX_META_TENSOR_NAME}vec.size());
{code_indent}  for (size_t i = 0; i < {PREFIX_OUTPUT}{out_name}_{PREFIX_META_TENSOR_NAME}vec.size(); ++i) {{
{code_indent}    {PREFIX_OUTPUT}{out_name}_metas[i] = {PREFIX_OUTPUT}{out_name}[i] ? &{PREFIX_OUTPUT}{out_name}_{PREFIX_META_TENSOR_NAME}vec[i] : nullptr;
{code_indent}  }}"""
                    )

                    param_code = param_code + out_name + '_metas, '
            else:
                meta_tensor_code = (
                    meta_tensor_code
                    + code_indent
                    + "  phi::MetaTensor "
                    + PREFIX_OUTPUT
                    + out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)
                    + "("
                    + PREFIX_OUTPUT
                    + out_name
                    + ");\n"
                )
                '''
                meta_tensor_code = (
                    meta_tensor_code
                    + "  "
                    + PREFIX_OUTPUT
                    + out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)
                    + f".share_meta({out_name});\n"
                )
                '''
                if len(kernel_output_names) == 1:
                    param_code = (
                        param_code
                        + f"&{PREFIX_OUTPUT}{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)}, "
                    )
                else:
                    param_code = (
                        param_code
                        + f"{PREFIX_OUTPUT}{out_name} ? &{PREFIX_OUTPUT}{out_name.replace('kernel_', PREFIX_META_TENSOR_NAME)} : nullptr, "
                    )

        param_code = param_code[:-2]
        return f"""{meta_tensor_code}
{code_indent}  phi::{infer_meta['func']}({param_code});
"""

    def gene_debug_kernel(
        self, kernel_name, kernel_signature, kernel_args, outputs_args
    ):
        if self.api in SKIP_LIST:
            return ""

        code_indent = "  "
        # kernel_args_tmp = re.sub(r"input\_([a-z_]+)", r"\1", kernel_args)
        kernel_args_tmp = re.sub(r"\*?input_([a-z]+)", r"\1", kernel_args)
        kernel_args_tmp = kernel_args_tmp.split(", ")
        kernel_args = kernel_args.split(", ")
        for idx, arg in enumerate(kernel_args_tmp):
            if arg in self.inputs['names']:
                kernel_args[idx] = kernel_args[idx].replace(
                    "input", "debug_input", 1
                )
        kernel_args = ", ".join(kernel_args)

        kernel_args = kernel_args.replace("dev_ctx", "debug_dev_ctx")
        outputs_args = list(
            map(lambda s: s.replace("kernel_", "debug_kernel_"), outputs_args)
        )

        code = ""
        '''
        debug_kernel_args = kernel_args.replace("*", "")
        debug_kernel_args = debug_kernel_args.split(", ")
        code = f"{code_indent}  OpOutputDebugger::PrintOutput({debug_kernel_args[1]}, Backend::CPU);" if debug_kernel_args[1].startswith("debug_input") else f"std::cout << \"{debug_kernel_args[1]}\" << std::endl;"
        if len(debug_kernel_args) > 2:
            code += f"{code_indent}  OpOutputDebugger::PrintOutput({debug_kernel_args[2]}, Backend::CPU);" if debug_kernel_args[2].startswith("debug_input") else f"std::cout << \"{debug_kernel_args[2]}\" << std::endl;"
        '''
        return (
            code
            + f"""
{code_indent}if (std::getenv("XPU_DY_ACC_DEBUG_DIFF") != nullptr) {{
{code_indent}  auto {PREFIX_OUTPUT}kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
{code_indent}      "{kernel_name}", {{Backend::CPU, kernel_layout, kernel_data_type}});
{code_indent}  const auto& {PREFIX_OUTPUT}kernel = {PREFIX_OUTPUT}kernel_result.kernel;
{code_indent}  auto* {PREFIX_OUTPUT}dev_ctx = GetDeviceContextByBackend(Backend::CPU);
{code_indent}  using {PREFIX_OUTPUT}kernel_signature = {kernel_signature};
{code_indent}  auto* {PREFIX_OUTPUT}kernel_fn = {PREFIX_OUTPUT}kernel.GetVariadicKernelFn<{PREFIX_OUTPUT}kernel_signature>();
{code_indent}  (*{PREFIX_OUTPUT}kernel_fn)({kernel_args}, {", ".join(outputs_args)});
{code_indent}}}"""
        )

    def gene_api_declaration(self):
        if not self.is_base_api:
            invoke_func_name = self.invoke.split('(')[0]
            if (not invoke_func_name.endswith("_grad")) and (
                not invoke_func_name.endswith('_impl')
            ):
                return ""
        api_func_name = self.get_api_func_name()
        api_declaration = f"""
PADDLE_API void {api_func_name}({self.get_declare_args()});
"""
        return api_declaration

    def gene_kernel_backend_select(self):
        all_no_need_buffer = True
        for in_name in self.inputs['names']:
            if in_name not in self.no_need_buffer:
                all_no_need_buffer = False

        if all_no_need_buffer:
            return """
  kernel_backend = ParseBackend(egr::Controller::Instance().GetExpectedPlace());
"""
        else:
            return super().gene_kernel_backend_select()

    def get_return_type(self, inplace_flag=False):
        return 'void'

    def gene_output(
        self,
        out_dtype_list,
        out_tensor_type_list=None,
        code_indent='',
        inplace_flag=False,
    ):
        kernel_output = []
        output_names = []
        output_create = ""

        if len(out_dtype_list) == 1:
            kernel_output.append('kernel_out')
            output_names.append('kernel_out')
            inplace_assign = (
                " = " + self.inplace_map[self.outputs['names'][0]]
                if inplace_flag
                and self.inplace_map is not None
                and self.outputs['names'][0] in self.inplace_map
                else ""
            )
            output_create = ""
            set_out_func = (
                'SetKernelOutput'
                if out_tensor_type_list is None
                or out_tensor_type_list[0] == 'dense'
                else 'SetSelectedRowsKernelOutput'
            )
            if out_dtype_list[0] == 'std::vector<Tensor>':
                assert (
                    self.outputs['out_size_expr'] is not None
                ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}(&{self.outputs['names'][0]});"""
                )

            else:
                output_create = (
                    output_create
                    + f"""
{code_indent}  auto kernel_out = {set_out_func}({self.outputs['names'][0]});"""
                )

        elif len(out_dtype_list) > 1:
            output_create = ""
            for i, out_type_item in enumerate(out_dtype_list):
                kernel_output.append(f'kernel_out_{i}')
                output_names.append(f'kernel_out_{i}')
                set_out_func = (
                    'SetKernelOutput'
                    if out_tensor_type_list is None
                    or out_tensor_type_list[i] == 'dense'
                    else 'SetSelectedRowsKernelOutput'
                )
                if out_type_item == 'Tensor':
                    if (
                        inplace_flag
                        and self.inplace_map is not None
                        and self.outputs['names'][i] in self.inplace_map
                    ):
                        output_create = (
                            output_create
                            + f"""
{code_indent}  *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""
                        )

                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}({self.outputs['names'][i]});"""
                    )

                else:
                    if (
                        inplace_flag
                        and self.inplace_map is not None
                        and self.outputs['names'][i] in self.inplace_map
                    ):
                        output_create = (
                            output_create
                            + f"""
{code_indent}  *{self.outputs['names'][i]} = {self.inplace_map[self.outputs['names'][i]]};"""
                        )

                    assert (
                        self.outputs['out_size_expr'][i] is not None
                    ), f"{self.api}: The out size expr : '{{expr}}' should be set when output has Tensor[]. You can refer 'split' api."
                    output_create = (
                        output_create
                        + f"""
{code_indent}  auto kernel_out_{i} = {set_out_func}(&{self.outputs['names'][i]});"""
                    )
        else:
            raise ValueError(
                "{} : Output error: the output should not be empty.".format(
                    self.api
                )
            )

        inplace = "true" if inplace_flag else "false"
        output_create += f"""
{code_indent}  bool inplace = {inplace};"""

        self.kernel_outputs = kernel_output

        return kernel_output, output_names, output_create

    def gene_invoke_code(self, invoke_code, params_code):
        invoke_func_name = invoke_code.split('(')[0].strip()
        if invoke_func_name.endswith('_grad') or invoke_func_name.endswith(
            '_impl'
        ):
            return f"""
PADDLE_API {self.get_return_type()} {self.api}({params_code}) {{
  {invoke_code};
}}"""

        else:
            return ""


def header_include():
    return """
#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"
"""


def source_include(header_file_path):
    return f"""
#include "{header_file_path}"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/api/include/api.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/unary.h"

#include "paddle/phi/api/profiler/event_tracing.h"
#include "paddle/phi/api/profiler/supplement_tracing.h"

#include "paddle/phi/api/lib/op_debug.h"
#include "paddle/fluid/platform/place.h"

DECLARE_bool(conv2d_disable_cudnn);
DECLARE_int32(low_precision_op_list);
"""


def backward_api_namespace():
    return (
        """
namespace paddle {
namespace experimental {

""",
        """

}  // namespace experimental
}  // namespace paddle
""",
    )


def op_global_id():
    return "static int global_id = 0;"


def generate_backward_api(
    backward_yaml_path, header_file_path, source_file_path
):

    bw_apis = []
    for each_api_yaml in backward_yaml_path:
        with open(each_api_yaml, 'r') as f:
            api_list = yaml.load(f, Loader=yaml.FullLoader)
            if api_list:
                bw_apis.extend(api_list)

    header_file = open(header_file_path, 'w')
    source_file = open(source_file_path, 'w')

    namespace = backward_api_namespace()
    global_id = op_global_id()

    header_file.write("#pragma once\n")
    header_file.write(header_include())
    header_file.write(namespace[0])

    include_header_file = "paddle/phi/api/backward/backward_api.h"
    source_file.write(source_include(include_header_file))
    source_file.write(namespace[0])
    source_file.write(global_id)

    for bw_api in bw_apis:
        bw_api = BackwardAPI(bw_api)
        header_file.write(bw_api.gene_api_declaration())
        source_file.write(bw_api.gene_api_code())

    header_file.write(namespace[1])
    source_file.write(namespace[1])

    header_file.close()
    source_file.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate PaddlePaddle C++ backward API files'
    )
    parser.add_argument(
        '--backward_yaml_path',
        help='path to backward yaml file',
        nargs='+',
        default=['paddle/phi/api/yaml/backward.yaml'],
    )
    parser.add_argument(
        '--backward_header_path',
        help='output of generated backward header code file',
        default='paddle/phi/api/backward/backward_api.h',
    )

    parser.add_argument(
        '--backward_source_path',
        help='output of generated backward source code file',
        default='paddle/phi/api/lib/backward_api.cc',
    )

    options = parser.parse_args()

    backward_yaml_path = options.backward_yaml_path
    header_file_path = options.backward_header_path
    source_file_path = options.backward_source_path

    generate_backward_api(
        backward_yaml_path, header_file_path, source_file_path
    )


if __name__ == '__main__':
    main()
