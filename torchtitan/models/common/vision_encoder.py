# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared model-agnostic ViT building blocks for VLM vision encoders: a
block-diagonal FlexAttention mask helper and the pre-norm transformer block
(attention + MLP) over token-major visual patches.

RoPE differs per model, so each encoder passes it through the block to the
attention as two per-forward args: ``rope_cache`` (a tensor, so config-based
sharding can DTensor-wrap it before it meets the head-sharded q/k) and
``rope_apply`` (a pass-through callable ``(q, k, rope_cache) -> (q, k)``).

Shape suffixes:
- T = packed visual tokens
- D = vision dim
- H = num heads
- Dh = head dim
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention, local_head_split
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm
from torchtitan.protocols.module import Module

compiled_create_block_mask = torch.compile(create_block_mask)

# Applies rotary position embedding: (query, key, rope_cache) -> (query, key).
RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def create_block_diagonal_mask(
    segment_lengths: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> BlockMask:
    """Create a FlexAttention mask over contiguous packed segments."""
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.shape[0], device=device, dtype=torch.int32),
        segment_lengths.to(device=device, dtype=torch.int32),
        # Avoid reading segment_lengths.sum() back to the host to size the
        # output; the packed token count is already available from its shape.
        output_size=total_tokens,
    )

    def mask_mod(b, h, q_idx, kv_idx):
        return segment_ids[q_idx] == segment_ids[kv_idx]

    return compiled_create_block_mask(
        mask_mod,
        1,
        None,
        total_tokens,
        total_tokens,
        device=device,
    )


def create_block_diagonal_sdpa_mask(
    segment_lengths: torch.Tensor,
    total_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Create an exact dense block-diagonal mask for packed visual segments."""
    segment_ids_T = torch.repeat_interleave(
        torch.arange(segment_lengths.shape[0], device=device, dtype=torch.int32),
        segment_lengths.to(device=device, dtype=torch.int32),
        output_size=total_tokens,
    )
    return segment_ids_T[:, None] == segment_ids_T[None, :]


class VisionScaledDotProductAttention(Module):
    """Non-causal SDPA over packed visual segments with a dense segment mask."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    sdpa_backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        q_TNH: torch.Tensor,
        k_TNH: torch.Tensor,
        v_TNH: torch.Tensor,
        *,
        attention_masks: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        q_NTH, k_NTH, v_NTH = (
            q_TNH.transpose(0, 1),
            k_TNH.transpose(0, 1),
            v_TNH.transpose(0, 1),
        )
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out_NTH = F.scaled_dot_product_attention(
                q_NTH,
                k_NTH,
                v_NTH,
                attn_mask=attention_masks,
                scale=scale,
                is_causal=False,
            )
        return out_NTH.transpose(0, 1)


class VisionMLP(Module):
    """Feed-forward network with GELU activation (fc1 -> act -> fc2)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )

    def __init__(self, config: Config):
        super().__init__()
        self.linear_fc1 = config.fc1.build()
        self.linear_fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(Module):
    """Multi-head self-attention over visual patches.

    Separate q/k/v projections (clean per-head ColwiseParallel under TP). RoPE is
    applied via the injected ``rope_apply`` callable so this class is reused
    across models with different rotary formulations.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError(
                f"VisionAttention dim ({config.dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.dim // config.num_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask | torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]

        # -1 infers the head count locally (= num_heads / TP under tensor
        # parallelism, where wq/wk/wv are colwise-sharded).
        q_THDh = local_head_split(self.wq(x), self.head_dim)
        k_THDh = local_head_split(self.wk(x), self.head_dim)
        v_THDh = local_head_split(self.wv(x), self.head_dim)

        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)

        out_THDh = self.flex_attention(
            q_THDh, k_THDh, v_THDh, attention_masks=attention_mask
        )
        out_TD = out_THDh.reshape(num_tokens, -1)
        return self.proj(out_TD)


class VisionTransformerBlock(Module):
    """Pre-norm transformer block: norm -> attn -> residual -> norm -> mlp."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # MoonViT normalizes with RMSNorm; Qwen3.5 and Muse Glimmer use LayerNorm.
        norm1: LayerNorm.Config | RMSNorm.Config
        norm2: LayerNorm.Config | RMSNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask | torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x
