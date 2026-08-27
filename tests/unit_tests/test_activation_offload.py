# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit coverage for selective pinned activation offload (P1)."""

import copy
import unittest

import torch
import torch.nn as nn

from torchtitan.config import ActivationOffloadConfig
from torchtitan.distributed.activation_offload import (
    OffloadedActivation,
    SelectiveOffloadPolicy,
    apply_activation_offload,
)


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)) * x + residual.sum()


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _ToyBlock(), "1": _ToyBlock()})

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x, residual)
        return x


class TestSelectiveOffloadPolicy(unittest.TestCase):
    def test_policy_filters_device_size_storage_and_offset(self):
        base = torch.zeros(1024, 1024)
        policy = SelectiveOffloadPolicy(
            device_type="cpu",
            min_tensor_bytes=1024,
            max_tensor_bytes=None,
            excluded_storage_ptrs={base.untyped_storage().data_ptr()},
        )
        self.assertFalse(policy.allows(base))

        policy = SelectiveOffloadPolicy(
            device_type="cpu",
            min_tensor_bytes=1024,
            max_tensor_bytes=None,
            excluded_storage_ptrs=set(),
        )
        self.assertFalse(policy.allows(torch.zeros(1, 1)))
        self.assertTrue(policy.allows(torch.zeros(512, 512)))
        self.assertFalse(policy.allows(torch.zeros(3, 3)[1:]))

    def test_selective_offload_preserves_cpu_output_and_gradients(self):
        torch.manual_seed(0)
        reference = _ToyDecoder()
        offloaded = copy.deepcopy(reference)
        apply_activation_offload(
            offloaded,
            config=ActivationOffloadConfig(
                pin_memory=False,
                device_type="cpu",
                min_tensor_bytes=1,
            ),
        )

        reference_input = torch.randn(2, 4, 8, requires_grad=True)
        offloaded_input = reference_input.detach().clone().requires_grad_(True)
        reference_residual = torch.randn(2, 4, 8)
        offloaded_residual = reference_residual.clone()

        reference_output = reference(reference_input, reference_residual)
        offloaded_output = offloaded(offloaded_input, offloaded_residual)
        torch.testing.assert_close(offloaded_output, reference_output)

        reference_output.square().mean().backward()
        offloaded_output.square().mean().backward()
        torch.testing.assert_close(offloaded_input.grad, reference_input.grad)

        for reference_parameter, offloaded_parameter in zip(
            reference.parameters(), offloaded.parameters(), strict=True
        ):
            torch.testing.assert_close(
                offloaded_parameter.grad, reference_parameter.grad
            )

    def test_offloaded_activation_round_trip(self):
        source = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        packed = OffloadedActivation(source, pin=False)
        restored = packed.restore()
        torch.testing.assert_close(restored, source)

    def test_rejects_double_wrapping(self):
        model = _ToyDecoder()
        config = ActivationOffloadConfig(
            pin_memory=False, device_type="cpu", min_tensor_bytes=1
        )
        apply_activation_offload(model, config=config)
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            apply_activation_offload(model, config=config)

    def test_config_defaults(self):
        config = ActivationOffloadConfig()
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.device_type, "npu")
        self.assertEqual(config.min_tensor_bytes, 1 << 20)
        self.assertIsNone(config.max_tensor_bytes)


if __name__ == "__main__":
    unittest.main()
