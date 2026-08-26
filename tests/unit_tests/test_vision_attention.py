# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_sdpa_mask,
    VisionAttention,
    VisionScaledDotProductAttention,
)


class _IdentityAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: torch.Size | None = None

    def forward(self, q_TNH, k_TNH, v_TNH, *, attention_masks):
        self.input_shape = q_TNH.shape
        return q_TNH


class TestVisionAttention(unittest.TestCase):
    def test_td_layout(self) -> None:
        dim, num_heads, num_tokens = 16, 4, 7
        attention = VisionAttention(
            VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wk=Linear.Config(in_features=dim, out_features=dim),
                wv=Linear.Config(in_features=dim, out_features=dim),
                proj=Linear.Config(in_features=dim, out_features=dim),
            )
        )
        identity_attention = _IdentityAttention()
        attention.flex_attention = identity_attention

        x_TD = torch.randn(num_tokens, dim)
        attention_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: torch.tensor(True),
            B=1,
            H=None,
            Q_LEN=num_tokens,
            KV_LEN=num_tokens,
            device="cpu",
        )

        out_TD = attention(
            x_TD,
            rope_cache=torch.empty(0),
            rope_apply=lambda q, k, cache: (q, k),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))

    def test_sdpa_respects_packed_segments(self) -> None:
        torch.manual_seed(0)
        attention = VisionScaledDotProductAttention(
            VisionScaledDotProductAttention.Config()
        )
        q_TNH = torch.randn(4, 2, 3)
        k_TNH = torch.randn(4, 2, 3)
        v_TNH = torch.randn(4, 2, 3)
        attention_mask_TT = create_block_diagonal_sdpa_mask(
            torch.tensor([2, 2]), total_tokens=4, device=torch.device("cpu")
        )

        out_TNH = attention(
            q_TNH, k_TNH, v_TNH, attention_masks=attention_mask_TT
        )
        changed_v_TNH = v_TNH.clone()
        changed_v_TNH[2:] += 100
        changed_out_TNH = attention(
            q_TNH, k_TNH, changed_v_TNH, attention_masks=attention_mask_TT
        )

        torch.testing.assert_close(out_TNH[:2], changed_out_TNH[:2])
        self.assertFalse(torch.allclose(out_TNH[2:], changed_out_TNH[2:]))


if __name__ == "__main__":
    unittest.main()
