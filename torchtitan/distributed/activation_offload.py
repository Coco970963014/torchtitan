# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selective saved-activation offload with pinned host buffers (P1)."""

import functools

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


class OffloadedActivation:
    """Pinned host copy of one saved tensor, restored on demand."""

    __slots__ = ("cpu_tensor", "device")

    def __init__(self, tensor: torch.Tensor, *, pin: bool) -> None:
        self.cpu_tensor = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            pin_memory=pin,
        )
        self.cpu_tensor.copy_(tensor, non_blocking=pin)
        self.device = tensor.device

    def restore(self) -> torch.Tensor:
        restored = torch.empty(
            self.cpu_tensor.size(),
            dtype=self.cpu_tensor.dtype,
            device=self.device,
        )
        restored.copy_(self.cpu_tensor, non_blocking=True)
        return restored


class SelectiveOffloadPolicy:
    """Decide whether an autograd-saved tensor may be copied to host memory."""

    def __init__(
        self,
        *,
        device_type: str,
        min_tensor_bytes: int,
        max_tensor_bytes: int | None,
        excluded_storage_ptrs: set[int],
    ) -> None:
        self.device_type = device_type
        self.min_tensor_bytes = min_tensor_bytes
        self.max_tensor_bytes = max_tensor_bytes
        self.excluded_storage_ptrs = excluded_storage_ptrs

    def allows(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type != self.device_type:
            return False
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes < self.min_tensor_bytes:
            return False
        if self.max_tensor_bytes is not None and tensor_bytes > self.max_tensor_bytes:
            return False
        if tensor.untyped_storage().data_ptr() in self.excluded_storage_ptrs:
            return False
        if tensor.storage_offset() != 0:
            return False
        return True


def _input_storage_ptrs(args, kwargs) -> set[int]:
    ptrs = set()
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.untyped_storage().size() > 0:
            ptrs.add(value.untyped_storage().data_ptr())
    return ptrs


def _wrap_module_forward(module: nn.Module, *, config) -> None:
    if hasattr(module, "_activation_offload_original_forward"):
        raise RuntimeError(f"Activation offload is already applied to {module!r}.")

    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        policy = SelectiveOffloadPolicy(
            device_type=config.device_type,
            min_tensor_bytes=config.min_tensor_bytes,
            max_tensor_bytes=config.max_tensor_bytes,
            excluded_storage_ptrs=_input_storage_ptrs(args, kwargs),
        )

        def pack(tensor: torch.Tensor):
            if not policy.allows(tensor):
                return tensor
            return OffloadedActivation(tensor, pin=config.pin_memory)

        def unpack(packed):
            if isinstance(packed, OffloadedActivation):
                return packed.restore()
            return packed

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            return original_forward(*args, **kwargs)

    module._activation_offload_original_forward = original_forward
    module.forward = wrapped_forward


def apply_activation_offload(model: nn.Module, *, config) -> None:
    """Apply selective pinned offload to every decoder transformer block."""
    layers = model.get_submodule("layers")
    wrapped_layers = []
    for layer_id, transformer_block in layers.named_children():
        _wrap_module_forward(transformer_block, config=config)
        wrapped_layers.append(layer_id)

    logger.info(
        "Applied selective pinned saved-tensor CPU offload to transformer blocks %s "
        "(device_type=%s, min_tensor_bytes=%s, max_tensor_bytes=%s, pin_memory=%s)",
        wrapped_layers,
        config.device_type,
        config.min_tensor_bytes,
        config.max_tensor_bytes,
        config.pin_memory,
    )
