# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public-autograd activation offload helpers for eager decoder blocks."""

import functools

import torch
import torch.nn as nn

from torchtitan.tools.logging import logger


def _wrap_module_forward(
    module: nn.Module,
    *,
    pin_memory: bool,
    device_type: str,
) -> None:
    if hasattr(module, "_activation_offload_original_forward"):
        raise RuntimeError(f"Activation offload is already applied to {module!r}.")

    original_forward = module.forward

    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        # save_on_cpu returns independent host copies and never mutates the
        # source tensor storage, which keeps cross-block aliases safe in P0.
        with torch.autograd.graph.save_on_cpu(
            pin_memory=pin_memory,
            device_type=device_type,
        ):
            return original_forward(*args, **kwargs)

    module._activation_offload_original_forward = original_forward
    module.forward = wrapped_forward


def apply_activation_offload(model: nn.Module, *, config) -> None:
    """Offload saved tensors created inside each decoder transformer block."""
    layers = model.get_submodule("layers")
    wrapped_layers = []
    for layer_id, transformer_block in layers.named_children():
        _wrap_module_forward(
            transformer_block,
            pin_memory=config.pin_memory,
            device_type=config.device_type,
        )
        wrapped_layers.append(layer_id)

    logger.info(
        "Applied public saved-tensor CPU offload to transformer blocks %s "
        "(device_type=%s, pin_memory=%s)",
        wrapped_layers,
        config.device_type,
        config.pin_memory,
    )
