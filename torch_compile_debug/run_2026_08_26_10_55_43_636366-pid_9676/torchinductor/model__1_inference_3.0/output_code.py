# AOT ID: ['1_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused_arange_0 = async_compile.cpp_pybinding(['int64_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(int64_t* out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(4L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(4L)))
                {
                    auto tmp0 = x0;
                    auto tmp1 = c10::convert<int64_t>(tmp0);
                    auto tmp2 = at::vec::VectorizedN<int64_t,2>::arange(tmp1, 1);
                    tmp2.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                }
            }
        }
    }
}
''')


cpp_fused_arange_bitwise_and_ge_view_1 = async_compile.cpp_pybinding(['const int32_t*', 'bool*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int32_t* in_ptr0,
                       bool* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(4L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(4L)))
                    {
                        auto tmp12 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = x1;
                        auto tmp3 = c10::convert<int64_t>(tmp2);
                        auto tmp4 = at::vec::VectorizedN<int64_t,2>::arange(tmp3, 1);
                        auto tmp5 = at::vec::VectorizedN<int64_t,2>(tmp1);
                        auto tmp6 = at::vec::VecMask<int64_t,2>(tmp5 >= tmp4);
                        auto tmp7 = static_cast<bool>(true);
                        auto tmp8 = at::vec::VecMask<float,1>::from(tmp7);
                        auto tmp9 = tmp8.template cast<int32_t,1>();
                        auto tmp10 = tmp6.template cast<int32_t,1>();
                        auto tmp11 = tmp9 & tmp10;
                        auto tmp13 = c10::convert<int64_t>(tmp12);
                        auto tmp14 = at::vec::VectorizedN<int64_t,2>(tmp13);
                        auto tmp15 = at::vec::VecMask<int64_t,2>(tmp4 >= tmp14);
                        auto tmp16 = tmp11.template cast<int32_t,1>();
                        auto tmp17 = tmp15.template cast<int32_t,1>();
                        auto tmp18 = tmp16 & tmp17;
                        tmp18.store(out_ptr0 + static_cast<int64_t>(x1 + 4L*x0), static_cast<int64_t>(4));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__to_copy_bitwise_and_gt_lt_permute_sum_view_2 = async_compile.cpp_pybinding(['const bool*', 'int64_t*', 'int32_t*', 'int32_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const bool* in_ptr0,
                       int64_t* out_ptr0,
                       int32_t* out_ptr1,
                       int32_t* out_ptr2)
{
    {
        {
            int64_t tmp_acc0 = 0;
            at::vec::VectorizedN<int64_t,2> tmp_acc0_vec = at::vec::VectorizedN<int64_t,2>(0);
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(16384L); x0+=static_cast<int64_t>(4L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16384L)))
                    {
                        auto tmp0 = at::vec::VecMask<float,1>::from(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(4));
                        auto tmp1 = tmp0.to<int64_t,2>();
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                }
            }
            tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<int64_t, 2>([](at::vec::Vectorized<int64_t>& x, at::vec::Vectorized<int64_t>& y) { return x + y; }, tmp_acc0_vec);
            out_ptr0[static_cast<int64_t>(0L)] = static_cast<int64_t>(tmp_acc0);
        }
    }
    {
        {
            {
                auto tmp0 = out_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int64_t>(0);
                auto tmp2 = tmp0 > tmp1;
                auto tmp3 = static_cast<int64_t>(16384);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = decltype(tmp2)(tmp2 & tmp4);
                auto tmp6 = c10::convert<int8_t>(tmp5);
                auto tmp7 = c10::convert<int32_t>(tmp6);
                auto tmp8 = c10::convert<int64_t>(tmp7);
                auto tmp9 = c10::convert<int32_t>(tmp8);
                out_ptr1[static_cast<int64_t>(0L)] = tmp7;
                out_ptr2[static_cast<int64_t>(0L)] = tmp9;
            }
        }
    }
}
''')


cpp_fused__to_copy_eq_3 = async_compile.cpp_pybinding(['const int64_t*', 'int32_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       int32_t* out_ptr0)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = static_cast<int64_t>(16384);
                auto tmp2 = tmp0 == tmp1;
                auto tmp3 = c10::convert<int8_t>(tmp2);
                auto tmp4 = c10::convert<int32_t>(tmp3);
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
            }
        }
    }
}
''')


cpp_fused__to_copy_arange_lt_new_zeros_scalar_tensor_unsqueeze_view_where_4 = async_compile.cpp_pybinding(['const int64_t*', 'const int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int64_t*', 'int64_t*', 'int32_t*', 'int32_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       const int32_t* in_ptr1,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       int32_t* out_ptr2,
                       int64_t* out_ptr3,
                       int64_t* out_ptr4,
                       int32_t* out_ptr5,
                       int32_t* out_ptr6)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp2 = in_ptr1[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int32_t>(tmp0);
                auto tmp3 = static_cast<int32_t>(0);
                auto tmp4 = tmp3 < tmp2;
                auto tmp5 = static_cast<int32_t>(1);
                auto tmp6 = tmp4 ? tmp1 : tmp5;
                out_ptr0[static_cast<int64_t>(0L)] = tmp1;
                out_ptr1[static_cast<int64_t>(0L)] = tmp6;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(4L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(2L)))
                {
                    auto tmp0 = static_cast<int32_t>(0);
                    auto tmp1 = at::vec::Vectorized<int32_t>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(2L));
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr3[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr4[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int32_t>(0);
                out_ptr5[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int32_t>(1);
                out_ptr6[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')


cpp_fused__to_copy_arange_lt_new_zeros_scalar_tensor_sum_unsqueeze_view_where_5 = async_compile.cpp_pybinding(['int32_t*', 'const int64_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int64_t*', 'int64_t*', 'int32_t*', 'int32_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(int32_t* in_out_ptr0,
                       const int64_t* in_ptr0,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       int32_t* out_ptr2,
                       int64_t* out_ptr3,
                       int64_t* out_ptr4,
                       int32_t* out_ptr5,
                       int32_t* out_ptr6)
{
    {
        {
            {
                auto tmp0 = in_out_ptr0[static_cast<int64_t>(0L)];
                auto tmp3 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int64_t>(tmp0);
                auto tmp2 = c10::convert<int32_t>(tmp1);
                auto tmp4 = c10::convert<int32_t>(tmp3);
                auto tmp5 = static_cast<int32_t>(0);
                auto tmp6 = tmp5 < tmp2;
                auto tmp7 = static_cast<int32_t>(1);
                auto tmp8 = tmp6 ? tmp4 : tmp7;
                in_out_ptr0[static_cast<int64_t>(0L)] = tmp2;
                out_ptr0[static_cast<int64_t>(0L)] = tmp4;
                out_ptr1[static_cast<int64_t>(0L)] = tmp8;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(4L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(2L)))
                {
                    auto tmp0 = static_cast<int32_t>(0);
                    auto tmp1 = at::vec::Vectorized<int32_t>(tmp0);
                    tmp1.store(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(2L));
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr3[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr4[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int32_t>(0);
                out_ptr5[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int32_t>(1);
                out_ptr6[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
}
''')


cpp_fused__to_copy_slice_sum_transpose_6 = async_compile.cpp_pybinding(['const int64_t*', 'const int32_t*', 'const int64_t*', 'const int32_t*', 'int32_t*', 'int32_t*', 'int32_t*', 'int32_t*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const int64_t* in_ptr0,
                       const int32_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const int32_t* in_ptr3,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       int32_t* out_ptr2,
                       int32_t* out_ptr3)
{
    {
        {
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int32_t>(tmp0);
                out_ptr0[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr1[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int64_t>(tmp0);
                auto tmp2 = c10::convert<int32_t>(tmp1);
                out_ptr1[static_cast<int64_t>(0L)] = tmp2;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr2[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int32_t>(tmp0);
                out_ptr2[static_cast<int64_t>(0L)] = tmp1;
            }
        }
    }
    {
        {
            {
                auto tmp0 = in_ptr3[static_cast<int64_t>(0L)];
                auto tmp1 = c10::convert<int64_t>(tmp0);
                auto tmp2 = c10::convert<int32_t>(tmp1);
                out_ptr3[static_cast<int64_t>(0L)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        buf0 = empty_strided_cpu((4, ), (1, ), torch.int64)
        assert_size_stride(arg0_1, (4, ), (1, ))
        cpp_fused_arange_0(buf0)
        # Topologically Sorted Source Nodes: [m, x], Original ATen: [aten.arange, aten.index]
        buf1 = torch.ops.aten.index.Tensor(arg0_1, [buf0])
        del arg0_1
        del buf0
        buf2 = buf1
        assert_size_stride(buf2, (4, ), (1, ), 'torch.ops.aten.index.Tensor')
        # buffer buf2 (op: torch.ops.aten.index.Tensor) is assumed to be not aligned
        del buf1
        assert_size_stride(arg1_1, (128, ), (1, ))
        # Topologically Sorted Source Nodes: [index_1], Original ATen: [aten.index]
        buf3 = torch.ops.aten.index.Tensor(arg1_1, [buf2])
        del arg1_1
        del buf2
        buf4 = buf3
        assert_size_stride(buf4, (4, ), (1, ), 'torch.ops.aten.index.Tensor')
        # buffer buf4 (op: torch.ops.aten.index.Tensor) is assumed to be not aligned
        del buf3
        buf5 = empty_strided_cpu((1, 4, 4), (16, 4, 1), torch.bool)
        cpp_fused_arange_bitwise_and_ge_view_1(buf4, buf5)
        del buf4
        # Topologically Sorted Source Nodes: [result_1, m, ge, n, ge_1, result_2, batched_outputs_2, mask_1], Original ATen: [aten.view, aten.arange, aten.ge, aten.bitwise_and, aten.constant_pad_nd]
        buf6 = torch.ops.aten.constant_pad_nd.default(reinterpret_tensor(buf5, (1, 1, 4, 4), (0, 0, 4, 1), 0), [0, 124, 0, 124], 0.0)
        del buf5
        buf7 = buf6
        assert_size_stride(buf7, (1, 1, 128, 128), (16384, 16384, 128, 1), 'torch.ops.aten.constant_pad_nd.default')
        assert_alignment(buf7, 16, 'torch.ops.aten.constant_pad_nd.default')
        del buf6
        buf8 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int64)
        buf9 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf17 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.int32)
        cpp_fused__to_copy_bitwise_and_gt_lt_permute_sum_view_2(buf7, buf8, buf9, buf17)
        del buf7
        # Topologically Sorted Source Nodes: [gt, lt, partial_blocks, partial_blocks_1, dense_mask, col_indices], Original ATen: [aten.gt, aten.lt, aten.bitwise_and, aten._to_copy, aten.sort]
        buf10 = torch.ops.aten.sort.stable(buf9, stable=True, descending=True)
        del buf9
        buf12 = buf10[1]
        assert_size_stride(buf12, (1, 1, 1, 1), (1, 1, 1, 1), 'torch.ops.aten.sort.stable')
        assert_alignment(buf12, 16, 'torch.ops.aten.sort.stable')
        del buf10
        buf13 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        cpp_fused__to_copy_eq_3(buf8, buf13)
        del buf8
        # Topologically Sorted Source Nodes: [full_blocks, full_blocks_1, dense_mask_1, col_indices_1], Original ATen: [aten.eq, aten._to_copy, aten.sort]
        buf14 = torch.ops.aten.sort.stable(buf13, stable=True, descending=True)
        buf16 = buf14[1]
        assert_size_stride(buf16, (1, 1, 1, 1), (1, 1, 1, 1), 'torch.ops.aten.sort.stable')
        assert_alignment(buf16, 16, 'torch.ops.aten.sort.stable')
        del buf14
        buf18 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf23 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf19 = empty_strided_cpu((1, 1, 1, 2), (2, 2, 2, 1), torch.int32)
        buf20 = empty_strided_cpu((1, ), (1, ), torch.int64)
        buf21 = empty_strided_cpu((1, ), (1, ), torch.int64)
        buf22 = empty_strided_cpu((1, ), (1, ), torch.int32)
        buf24 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        cpp_fused__to_copy_arange_lt_new_zeros_scalar_tensor_unsqueeze_view_where_4(buf12, buf17, buf18, buf23, buf19, buf20, buf21, buf22, buf24)
        del buf12
        # Topologically Sorted Source Nodes: [dense_mask_2, setitem, arange_4, row_indices, col_range, unsqueeze_1, index_mask, valid_indices], Original ATen: [aten.new_zeros, aten.arange, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
        buf25 = torch.ops.aten.index_put_.default(buf19, [reinterpret_tensor(buf20, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf21, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf22, (1, 1), (0, 0), 0), buf23], buf24)
        del buf20
        del buf21
        del buf22
        del buf23
        del buf24
        buf26 = buf25
        assert_size_stride(buf26, (1, 1, 1, 2), (2, 2, 2, 1), 'torch.ops.aten.index_put_.default')
        assert_alignment(buf26, 16, 'torch.ops.aten.index_put_.default')
        del buf19
        # Topologically Sorted Source Nodes: [num_blocks_in_row_2, col_indices_2], Original ATen: [aten.slice, aten.transpose, aten.sort]
        buf27 = torch.ops.aten.sort.stable(reinterpret_tensor(buf26, (1, 1, 1, 1), (2, 2, 1, 2), 0), stable=True, descending=True)
        buf29 = buf27[1]
        assert_size_stride(buf29, (1, 1, 1, 1), (1, 1, 1, 1), 'torch.ops.aten.sort.stable')
        assert_alignment(buf29, 16, 'torch.ops.aten.sort.stable')
        del buf27
        buf30 = reinterpret_tensor(buf13, (1, 1, 1), (1, 1, 1), 0); del buf13  # reuse
        buf31 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf36 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf32 = empty_strided_cpu((1, 1, 1, 2), (2, 2, 2, 1), torch.int32)
        buf33 = empty_strided_cpu((1, ), (1, ), torch.int64)
        buf34 = empty_strided_cpu((1, ), (1, ), torch.int64)
        buf35 = empty_strided_cpu((1, ), (1, ), torch.int32)
        buf37 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        cpp_fused__to_copy_arange_lt_new_zeros_scalar_tensor_sum_unsqueeze_view_where_5(buf30, buf16, buf31, buf36, buf32, buf33, buf34, buf35, buf37)
        del buf16
        # Topologically Sorted Source Nodes: [dense_mask_4, setitem_1, arange_6, row_indices_1, col_range_1, unsqueeze_3, index_mask_1, valid_indices_1], Original ATen: [aten.new_zeros, aten.arange, aten.unsqueeze, aten.lt, aten.scalar_tensor, aten.where, aten.view, aten.index_put]
        buf38 = torch.ops.aten.index_put_.default(buf32, [reinterpret_tensor(buf33, (1, 1, 1, 1), (0, 0, 0, 0), 0), reinterpret_tensor(buf34, (1, 1, 1), (0, 0, 0), 0), reinterpret_tensor(buf35, (1, 1), (0, 0), 0), buf36], buf37)
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        buf39 = buf38
        assert_size_stride(buf39, (1, 1, 1, 2), (2, 2, 2, 1), 'torch.ops.aten.index_put_.default')
        assert_alignment(buf39, 16, 'torch.ops.aten.index_put_.default')
        del buf32
        # Topologically Sorted Source Nodes: [num_blocks_in_row_3, col_indices_3], Original ATen: [aten.slice, aten.transpose, aten.sort]
        buf40 = torch.ops.aten.sort.stable(reinterpret_tensor(buf39, (1, 1, 1, 1), (2, 2, 1, 2), 0), stable=True, descending=True)
        buf42 = buf40[1]
        assert_size_stride(buf42, (1, 1, 1, 1), (1, 1, 1, 1), 'torch.ops.aten.sort.stable')
        assert_alignment(buf42, 16, 'torch.ops.aten.sort.stable')
        del buf40
        buf43 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf44 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.int32)
        buf45 = empty_strided_cpu((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        buf46 = empty_strided_cpu((1, 1, 1), (1, 1, 1), torch.int32)
        cpp_fused__to_copy_slice_sum_transpose_6(buf42, buf39, buf29, buf26, buf43, buf44, buf45, buf46)
        return (buf43, buf44, buf45, buf46, buf31, buf30, buf18, buf17, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.int32)
    arg1_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.int32)
    return [arg0_1, arg1_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
