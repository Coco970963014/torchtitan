class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i32[4]", arg1_1: "i32[128]"):
        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1459 in create_mask, code: m = torch.arange(0, Q_LEN, device=device)
        iota_2: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1460 in create_mask, code: n = torch.arange(0, KV_LEN, device=device)
        iota_3: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view: "i64[4, 1]" = torch.ops.aten.view.default(iota_2, [4, 1])
        ge: "b8[4, 4]" = torch.ops.aten.ge.Tensor(view, iota_3);  view = None
        full_default: "b8[1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        bitwise_and: "b8[1, 4, 4]" = torch.ops.aten.bitwise_and.Tensor(full_default, ge);  full_default = ge = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:100 in forward, code: return torch.ops.aten.index(x, indices)
        index: "i32[4]" = torch.ops.aten.index.Tensor(arg0_1, [iota_2]);  arg0_1 = iota_2 = None
        index_1: "i32[4]" = torch.ops.aten.index.Tensor(arg1_1, [index]);  arg1_1 = index = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/_dynamo/_trace_wrapped_higher_order_op.py:158 in __torch_function__, code: return func(*args, **(kwargs or {}))
        view_2: "i32[4, 1]" = torch.ops.aten.view.default(index_1, [4, 1]);  index_1 = None
        ge_1: "b8[4, 4]" = torch.ops.aten.ge.Tensor(iota_3, view_2);  iota_3 = view_2 = None
        bitwise_and_1: "b8[1, 4, 4]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ge_1);  bitwise_and = ge_1 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/_functorch/vmap.py:204 in _maybe_remove_batch_dim, code: return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        view_3: "b8[1, 1, 4, 4]" = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 4, 4]);  bitwise_and_1 = None
        expand: "b8[1, 1, 4, 4]" = torch.ops.aten.expand.default(view_3, [1, 1, 4, 4]);  view_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "b8[1, 1, 128, 128]" = torch.ops.aten.constant_pad_nd.default(expand, [0, 124, 0, 124], 0.0);  expand = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1339 in _convert_mask_to_block_mask, code: mask = mask.view(
        view_4: "b8[1, 1, 1, 128, 1, 128]" = torch.ops.aten.view.default(constant_pad_nd, [1, 1, 1, 128, 1, 128]);  constant_pad_nd = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1342 in _convert_mask_to_block_mask, code: mask = mask.permute(
        permute: "b8[1, 1, 1, 1, 128, 128]" = torch.ops.aten.permute.default(view_4, [0, 1, 2, 4, 3, 5]);  view_4 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1345 in _convert_mask_to_block_mask, code: mask_block_sum = mask.sum(
        sum_1: "i64[1, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(permute, [-2, -1]);  permute = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1350 in _convert_mask_to_block_mask, code: full_blocks = mask_block_sum == full_block_sum
        eq: "b8[1, 1, 1, 1]" = torch.ops.aten.eq.Scalar(sum_1, 16384)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1351 in _convert_mask_to_block_mask, code: partial_blocks = (mask_block_sum > 0) & (mask_block_sum < full_block_sum)
        gt: "b8[1, 1, 1, 1]" = torch.ops.aten.gt.Scalar(sum_1, 0)
        lt: "b8[1, 1, 1, 1]" = torch.ops.aten.lt.Scalar(sum_1, 16384);  sum_1 = None
        bitwise_and_2: "b8[1, 1, 1, 1]" = torch.ops.aten.bitwise_and.Tensor(gt, lt);  gt = lt = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1352 in _convert_mask_to_block_mask, code: partial_blocks = partial_blocks.to(dtype=torch.int8)
        convert_element_type: "i8[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(bitwise_and_2, torch.int8);  bitwise_and_2 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:1353 in _convert_mask_to_block_mask, code: full_blocks = full_blocks.to(dtype=torch.int8)
        convert_element_type_1: "i8[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(eq, torch.int8);  eq = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:424 in _dense_to_ordered, code: dense_mask = dense_mask.to(dtype=torch.int32)
        convert_element_type_2: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.int32);  convert_element_type = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_2: "i64[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [-1])

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort = torch.ops.aten.sort.stable(convert_element_type_2, stable = True, descending = True);  convert_element_type_2 = None
        getitem_1: "i64[1, 1, 1, 1]" = sort[1];  sort = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_3: "i32[1, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_2, torch.int32);  sum_2 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_4: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.int32);  getitem_1 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:424 in _dense_to_ordered, code: dense_mask = dense_mask.to(dtype=torch.int32)
        convert_element_type_5: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.int32);  convert_element_type_1 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_3: "i64[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [-1])

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_1 = torch.ops.aten.sort.stable(convert_element_type_5, stable = True, descending = True);  convert_element_type_5 = None
        getitem_3: "i64[1, 1, 1, 1]" = sort_1[1];  sort_1 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_6: "i32[1, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_3, torch.int32);  sum_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_7: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.int32);  getitem_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_1: "i32[1, 1, 1, 2]" = torch.ops.aten.full.default([1, 1, 1, 2], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_4: "i32[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze: "i32[1, 1]" = torch.ops.aten.unsqueeze.default(iota_4, -1);  iota_4 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_5: "i32[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_1: "i32[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 3)
        lt_1: "b8[1, 1, 1, 1]" = torch.ops.aten.lt.Tensor(iota_5, unsqueeze_1);  iota_5 = unsqueeze_1 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_2: "i32[]" = torch.ops.aten.full.default([], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "i32[1, 1, 1, 1]" = torch.ops.aten.where.self(lt_1, convert_element_type_4, full_default_2);  lt_1 = full_default_2 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_6: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_2: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_6, -1);  iota_6 = None
        unsqueeze_3: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        full_default_3: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_7: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_4: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_7, -1);  iota_7 = None
        unsqueeze_5: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, -1);  unsqueeze_5 = None
        index_put: "i32[1, 1, 1, 2]" = torch.ops.aten.index_put.default(full_default_1, [unsqueeze_6, unsqueeze_3, unsqueeze, where], full_default_3);  full_default_1 = unsqueeze_6 = unsqueeze_3 = unsqueeze = where = full_default_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        slice_3: "i32[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(index_put, 3, 0, 1);  index_put = None
        permute_2: "i32[1, 1, 1, 1]" = torch.ops.aten.permute.default(slice_3, [0, 1, 3, 2]);  slice_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_4: "i64[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(permute_2, [-1])

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_2 = torch.ops.aten.sort.stable(permute_2, stable = True, descending = True);  permute_2 = None
        getitem_5: "i64[1, 1, 1, 1]" = sort_2[1];  sort_2 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_8: "i32[1, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_4, torch.int32);  sum_4 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_9: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_5, torch.int32);  getitem_5 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:400 in create_dense_one, code: dense_mask = kv_indices.new_zeros(num_rows, num_cols + 1, dtype=torch.int32)
        full_default_4: "i32[1, 1, 1, 2]" = torch.ops.aten.full.default([1, 1, 1, 2], 0, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:402 in create_dense_one, code: row_indices = torch.arange(num_rows, dtype=torch.int, device=device).unsqueeze(
        iota_8: "i32[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze_7: "i32[1, 1]" = torch.ops.aten.unsqueeze.default(iota_8, -1);  iota_8 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:405 in create_dense_one, code: col_range = torch.arange(num_cols, dtype=torch.int, device=device)
        iota_9: "i32[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:406 in create_dense_one, code: index_mask = col_range < kv_num_blocks.unsqueeze(-1)
        unsqueeze_8: "i32[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3)
        lt_2: "b8[1, 1, 1, 1]" = torch.ops.aten.lt.Tensor(iota_9, unsqueeze_8);  iota_9 = unsqueeze_8 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:409 in create_dense_one, code: valid_indices = torch.where(index_mask, kv_indices, num_cols)
        full_default_5: "i32[]" = torch.ops.aten.full.default([], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_1: "i32[1, 1, 1, 1]" = torch.ops.aten.where.self(lt_2, convert_element_type_7, full_default_5);  lt_2 = full_default_5 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:412 in create_dense_one, code: dense_mask[row_indices, valid_indices] = dense_mask.new_ones(())
        iota_10: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_9: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_10, -1);  iota_10 = None
        unsqueeze_10: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, -1);  unsqueeze_9 = None
        full_default_6: "i32[1, 1, 1, 1]" = torch.ops.aten.full.default([1, 1, 1, 1], 1, dtype = torch.int32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_11: "i64[1]" = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze_11: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(iota_11, -1);  iota_11 = None
        unsqueeze_12: "i64[1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, -1);  unsqueeze_11 = None
        unsqueeze_13: "i64[1, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        index_put_1: "i32[1, 1, 1, 2]" = torch.ops.aten.index_put.default(full_default_4, [unsqueeze_13, unsqueeze_10, unsqueeze_7, where_1], full_default_6);  full_default_4 = unsqueeze_13 = unsqueeze_10 = unsqueeze_7 = where_1 = full_default_6 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:437 in _transpose_ordered, code: return _dense_to_ordered(dense.transpose(-2, -1))
        slice_6: "i32[1, 1, 1, 1]" = torch.ops.aten.slice.Tensor(index_put_1, 3, 0, 1);  index_put_1 = None
        permute_4: "i32[1, 1, 1, 1]" = torch.ops.aten.permute.default(slice_6, [0, 1, 3, 2]);  slice_6 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:425 in _dense_to_ordered, code: num_blocks_in_row = dense_mask.sum(dim=-1)
        sum_5: "i64[1, 1, 1]" = torch.ops.aten.sum.dim_IntList(permute_4, [-1])

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:426 in _dense_to_ordered, code: col_indices = torch.argsort(dense_mask, dim=-1, descending=True, stable=True)
        sort_3 = torch.ops.aten.sort.stable(permute_4, stable = True, descending = True);  permute_4 = None
        getitem_7: "i64[1, 1, 1, 1]" = sort_3[1];  sort_3 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:428 in _dense_to_ordered, code: num_blocks_in_row.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_10: "i32[1, 1, 1]" = torch.ops.prims.convert_element_type.default(sum_5, torch.int32);  sum_5 = None

        # File: /workspace/.venv-e0/lib/python3.12/site-packages/torch/nn/attention/flex_attention.py:429 in _dense_to_ordered, code: col_indices.to(torch.int32, memory_format=torch.contiguous_format),
        convert_element_type_11: "i32[1, 1, 1, 1]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.int32);  getitem_7 = None
        return (convert_element_type_11, convert_element_type_10, convert_element_type_9, convert_element_type_8, convert_element_type_7, convert_element_type_6, convert_element_type_4, convert_element_type_3)
