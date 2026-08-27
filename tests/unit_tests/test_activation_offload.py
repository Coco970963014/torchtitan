# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit coverage for the public saved-tensor activation offload wrapper."""

import copy
import unittest

import torch
import torch.nn as nn

from torchtitan.config import ActivationOffloadConfig
from torchtitan.distributed.activation_offload import apply_activation_offload


class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)) * x


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _ToyBlock(), "1": _ToyBlock()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x


class TestActivationOffload(unittest.TestCase):
    def test_saved_tensor_offload_preserves_cpu_output_and_gradients(self):
        torch.manual_seed(0)
        reference = _ToyDecoder()
        offloaded = copy.deepcopy(reference)
        config = ActivationOffloadConfig(pin_memory=False, device_type="npu")
        apply_activation_offload(offloaded, config=config)

        reference_input = torch.randn(2, 4, 8, requires_grad=True)
        offloaded_input = reference_input.detach().clone().requires_grad_(True)

        reference_output = reference(reference_input)
        offloaded_output = offloaded(offloaded_input)
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

    def test_saved_tensor_offload_rejects_double_wrapping(self):
        model = _ToyDecoder()
        config = ActivationOffloadConfig(pin_memory=False, device_type="npu")
        apply_activation_offload(model, config=config)
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            apply_activation_offload(model, config=config)

    def test_activation_offload_config_defaults(self):
        config = ActivationOffloadConfig()
        self.assertTrue(config.pin_memory)
        self.assertEqual(config.device_type, "npu")


if __name__ == "__main__":
    unittest.main()
