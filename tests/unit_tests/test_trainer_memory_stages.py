# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Regression coverage for allocator phase-boundary placement."""

import inspect

from torchtitan.trainer import Trainer


def test_after_forward_memory_stage_precedes_chunked_loss() -> None:
    """Keep the forward observation outside ChunkedLossWrapper backward work."""
    source = inspect.getsource(Trainer.forward_backward_step)
    assert source.index("pred = model_parts[0]") < source.index(
        'self._log_memory_stage("after_forward")'
    ) < source.index("loss, _ = self.loss_fn")
