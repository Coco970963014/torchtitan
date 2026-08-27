
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'
os.environ['TORCHINDUCTOR_COMPREHENSIVE_PADDING'] = '0'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config

torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.max_autotune = False
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.fallback_random = True
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.wrap_inductor_compiled_regions = False
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.12.0+cpu
# torch cuda version: None
# torch git version: 7661cd9c6b841b62b7f411aa52ec51f05457263b


# torch.cuda.is_available()==False, no GPU info collected
torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.reset_table()

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()



    def forward(self, arg0_1, arg1_1):
        iota_2 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota_3 = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view = torch.ops.aten.view.default(iota_2, [4, 1])
        ge = torch.ops.aten.ge.Tensor(view, iota_3);  view = None
        full_default = torch.ops.aten.full.default([1, 1, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        bitwise_and = torch.ops.aten.bitwise_and.Tensor(full_default, ge);  full_default = ge = None
        index = torch.ops.aten.index.Tensor(arg0_1, [iota_2]);  arg0_1 = iota_2 = None
        index_1 = torch.ops.aten.index.Tensor(arg1_1, [index]);  arg1_1 = index = None
        view_2 = torch.ops.aten.view.default(index_1, [4, 1]);  index_1 = None
        ge_1 = torch.ops.aten.ge.Tensor(iota_3, view_2);  iota_3 = view_2 = None
        bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ge_1);  bitwise_and = ge_1 = None
        view_3 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 4, 4]);  bitwise_and_1 = None
        expand = torch.ops.aten.expand.default(view_3, [1, 1, 4, 4]);  view_3 = None
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(expand, [0, 124, 0, 124], 0.0);  expand = None
        view_4 = torch.ops.aten.view.default(constant_pad_nd, [1, 1, 1, 128, 1, 128]);  constant_pad_nd = None
        permute = torch.ops.aten.permute.default(view_4, [0, 1, 2, 4, 3, 5]);  view_4 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(permute, [-2, -1]);  permute = None
        eq = torch.ops.aten.eq.Scalar(sum_1, 16384)
        gt = torch.ops.aten.gt.Scalar(sum_1, 0)
        lt = torch.ops.aten.lt.Scalar(sum_1, 16384);  sum_1 = None
        bitwise_and_2 = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None
        convert_element_type = torch.ops.prims.convert_element_type.default(bitwise_and_2, torch.int8);  bitwise_and_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(eq, torch.int8);  eq = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(convert_element_type, torch.int32);  convert_element_type = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [-1])
        sort = torch.ops.aten.sort.stable(convert_element_type_2, stable = True, descending = True);  convert_element_type_2 = None
        getitem_1 = sort[1];  sort = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sum_2, torch.int32);  sum_2 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(getitem_1, torch.int32);  getitem_1 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.int32);  convert_element_type_1 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [-1])
        sort_1 = torch.ops.aten.sort.stable(convert_element_type_5, stable = True, descending = True);  convert_element_type_5 = None
        getitem_3 = sort_1[1];  sort_1 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(sum_3, torch.int32);  sum_3 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(getitem_3, torch.int32);  getitem_3 = None
        full_default_1 = torch.ops.aten.full.default([1, 1, 1, 2], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_4 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(iota_4, -1);  iota_4 = None
        iota_5 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type_3, 3)
        lt_1 = torch.ops.aten.lt.Tensor(iota_5, unsqueeze_1);  iota_5 = unsqueeze_1 = None
        full_default_2 = torch.ops.aten.full.default([], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(lt_1, convert_element_type_4, full_default_2);  lt_1 = full_default_2 = None
        iota_6 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(iota_6, -1);  iota_6 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        full_default_3 = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_7 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(unsqueeze_5, -1);  unsqueeze_5 = None
        index_put = torch.ops.aten.index_put.default(full_default_1, [unsqueeze_6, unsqueeze_3, unsqueeze, where], full_default_3);  full_default_1 = unsqueeze_6 = unsqueeze_3 = unsqueeze = where = full_default_3 = None
        slice_3 = torch.ops.aten.slice.Tensor(index_put, 3, 0, 1);  index_put = None
        permute_2 = torch.ops.aten.permute.default(slice_3, [0, 1, 3, 2]);  slice_3 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(permute_2, [-1])
        sort_2 = torch.ops.aten.sort.stable(permute_2, stable = True, descending = True);  permute_2 = None
        getitem_5 = sort_2[1];  sort_2 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(sum_4, torch.int32);  sum_4 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(getitem_5, torch.int32);  getitem_5 = None
        full_default_4 = torch.ops.aten.full.default([1, 1, 1, 2], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_8 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(iota_8, -1);  iota_8 = None
        iota_9 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        lt_2 = torch.ops.aten.lt.Tensor(iota_9, unsqueeze_8);  iota_9 = unsqueeze_8 = None
        full_default_5 = torch.ops.aten.full.default([], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_1 = torch.ops.aten.where.self(lt_2, convert_element_type_7, full_default_5);  lt_2 = full_default_5 = None
        iota_10 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(iota_10, -1);  iota_10 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, -1);  unsqueeze_9 = None
        full_default_6 = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_11 = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, -1);  unsqueeze_11 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        index_put_1 = torch.ops.aten.index_put.default(full_default_4, [unsqueeze_13, unsqueeze_10, unsqueeze_7, where_1], full_default_6);  full_default_4 = unsqueeze_13 = unsqueeze_10 = unsqueeze_7 = where_1 = full_default_6 = None
        slice_6 = torch.ops.aten.slice.Tensor(index_put_1, 3, 0, 1);  index_put_1 = None
        permute_4 = torch.ops.aten.permute.default(slice_6, [0, 1, 3, 2]);  slice_6 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(permute_4, [-1])
        sort_3 = torch.ops.aten.sort.stable(permute_4, stable = True, descending = True);  permute_4 = None
        getitem_7 = sort_3[1];  sort_3 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(sum_5, torch.int32);  sum_5 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(getitem_7, torch.int32);  getitem_7 = None
        return (convert_element_type_11, convert_element_type_10, convert_element_type_9, convert_element_type_8, convert_element_type_7, convert_element_type_6, convert_element_type_4, convert_element_type_3)

def load_args(reader):
    buf0 = reader.storage(None, 16, dtype_hint=torch.int32)
    reader.tensor(buf0, (4,), dtype=torch.int32, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 512, dtype_hint=torch.int32)
    reader.tensor(buf1, (128,), dtype=torch.int32, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)