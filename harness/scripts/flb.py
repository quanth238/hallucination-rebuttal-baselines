"""Qwen-compatible FLB score correction, ported from the pinned official kernel."""
import math
from contextlib import AbstractContextManager

import torch
from transformers import LogitsProcessor


class FirstLogitBoost(LogitsProcessor):
    def __init__(self, gamma=0.3, rate=0.05, beta=0.1):
        if gamma < 0 or rate < 0 or not 0 < beta <= 1:
            raise ValueError('FLB requires gamma/rate >= 0 and 0 < beta <= 1.')
        self.gamma, self.rate, self.beta = gamma, rate, beta
        self.first, self.step = None, 0

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1:
            raise ValueError('The initial port supports batch=1 and one beam.')
        if not torch.isfinite(scores).all():
            raise ValueError('Non-finite input logits.')
        if self.first is None:
            self.first = scores.detach().clone()
        weight = self.gamma * (1.0 - math.exp(-self.rate * self.step))
        cutoff = math.log(self.beta) + scores.max(dim=-1, keepdim=True).values
        corrected = (scores + weight * self.first).masked_fill(scores < cutoff, -float('inf'))
        self.step += 1
        return corrected


class FlbContext(AbstractContextManager):
    """Correct raw model logits before HF's EOS/suppression processors.

The released sampler applies FLB before its generation processors. A custom
HF LogitsProcessor runs after built-in processors and therefore cannot safely
cache their masked scores as the original first-step distribution.
"""
    def __init__(self, model, gamma=0.3, rate=0.05, beta=0.1):
        self.model = model
        self.correction = FirstLogitBoost(gamma=gamma, rate=rate, beta=beta)
        self.handle = None

    def __enter__(self):
        def correct(module, args, output):
            output.logits = self.correction(None, output.logits[:, -1, :])[:, None, :]
            return output
        self.handle = self.model.register_forward_hook(correct)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

