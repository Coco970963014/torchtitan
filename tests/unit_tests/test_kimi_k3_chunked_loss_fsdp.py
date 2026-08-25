# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Regression for Kimi K3 chunked loss across the final FSDP boundary."""

from unittest.mock import patch

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule, fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import Embedding

try:
    from torchtitan.models.kimi_k3 import (
        _feed_forward_config,
        _linear,
        _mla_config,
        _norm,
    )
    from torchtitan.models.kimi_k3.model import KimiK3Model, KimiK3TransformerBlock
    from torchtitan.models.kimi_k3.parallelize import parallelize_kimi_k3
except ModuleNotFoundError as exc:
    raise RuntimeError(f"Kimi K3 optional dependency unavailable: {exc.name}") from exc


def _cpu_model_config() -> KimiK3Model.Config:
    """Build a CPU-safe Kimi K3 shape without the accelerator-only KDA path."""
    dim = 16
    block = KimiK3TransformerBlock.Config(
        layer_id=0,
        attn_res_block_size=1,
        attention=_mla_config(
            dim=dim,
            num_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
        ),
        delta_attention=None,
        feed_forward=_feed_forward_config(dim=dim, hidden_dim=32),
        moe=None,
        attention_norm=_norm(dim),
        ffn_norm=_norm(dim),
        attention_res_norm=_norm(dim),
        attention_res_proj=_linear(dim, 1),
        ffn_res_norm=_norm(dim),
        ffn_res_proj=_linear(dim, 1),
    )
    return KimiK3Model.Config(
        dim=dim,
        vocab_size=32,
        tok_embeddings=Embedding.Config(
            num_embeddings=32,
            embedding_dim=dim,
            param_init={"weight": lambda parameter: nn.init.normal_(parameter, std=0.02)},
        ),
        layers=[block],
        norm=_norm(dim),
        lm_head=_linear(dim, 32),
        output_res_norm=_norm(dim),
        output_res_proj=_linear(dim, 1),
        vision_encoder=None,
    )


class TestKimiK3ChunkedLossFSDP(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_chunked_loss_uses_separate_final_fsdp_units(self):
        torch.manual_seed(3)
        config = _cpu_model_config()
        with torch.device("meta"):
            model = config.build()
        model.to_empty(device=self.device_type)
        model.init_states()

        parallelism = ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        )
        parallel_dims = ParallelDims.from_config(parallelism, world_size=1)
        with patch("torchtitan.distributed.parallel_dims.device_type", self.device_type):
            parallel_dims.build_mesh()
        model = parallelize_kimi_k3(
            model,
            parallel_dims=parallel_dims,
            training=TrainingConfig(
                local_batch_size=1,
                seq_len=8,
                steps=1,
                dtype="bfloat16",
            ),
            parallelism=parallelism,
            compile_config=CompileConfig(),
            ac_config=None,
            dump_folder="",
        )

        assert isinstance(model, KimiK3Model)
        assert model.norm is not None
        assert model.lm_head is not None
        self.assertIsInstance(model.norm, FSDPModule)
        self.assertIsInstance(model.lm_head, FSDPModule)
        self.assertIsNot(
            fully_shard.state(model.norm),
            fully_shard.state(model.lm_head),
        )

        hidden_input = torch.randn(
            1,
            8,
            config.dim,
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        hidden_states = model.norm(hidden_input)
        loss_fn = ChunkedLossWrapper(
            ChunkedLossWrapper.Config(
                num_chunks=4,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=32),
            )
        )
        loss_fn.set_lm_head(model.lm_head)
        labels = torch.tensor(
            [[2, 3, 4, 5, 6, 7, 8, 9]],
            dtype=torch.long,
            device=self.device_type,
        )

        loss, _ = loss_fn(hidden_states, labels, global_valid_tokens=8.0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(hidden_input.grad)
        self.assertIsNotNone(model.norm.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)
