"""Architecture adapters for the pinned VISTA and CausalLens releases.

VISTA keeps VSV + SLA; CausalLens follows its released Qwen2-VL adapter.
Weights, RoPE and KV caching use the installed Qwen2.5-VL implementation.
"""
from contextlib import AbstractContextManager
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention, apply_multimodal_rotary_pos_emb, repeat_kv,
)


def vision_range(model, input_ids):
    tokens = input_ids[0]
    start = (tokens == model.config.vision_start_token_id).nonzero().flatten()
    end = (tokens == model.config.vision_end_token_id).nonzero().flatten()
    if len(start) != 1 or len(end) != 1:
        raise ValueError('This experiment expects one image per prompt.')
    return int(start[0]), int(end[0])


def vista_direction(negative, positive):
    """Released rank-one PCA recipe, including its centering and sign convention."""
    difference = (positive - negative).reshape(1, -1).float().cpu()
    mean = difference.mean(0, keepdim=True)
    u, _, vh = torch.linalg.svd(difference - mean, full_matrices=False)
    indices = torch.argmax(torch.abs(u), 0)
    signs = torch.sign(u[indices, torch.arange(u.shape[1])])
    component = vh * signs.view(-1, 1)
    return (component[:1].sum(0, keepdim=True) + mean).mean(0).view_as(positive)


def vista_steer(hidden, direction, strength=0.17):
    dtype = hidden.dtype
    x = hidden.float()
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    factor = 1 + F.cosine_similarity(x, -direction[None, None, :], dim=-1).clamp_min(0).unsqueeze(-1)
    shift = strength * factor * F.normalize(direction, dim=-1)[None, None, :]
    return (F.normalize(F.normalize(x, p=2, dim=-1) + shift, p=2, dim=-1) * norm).to(dtype)


class Vista(AbstractContextManager):
    def __init__(self, model, inputs, strength=0.17, alpha=0.3):
        self.model, self.inputs, self.strength, self.alpha = model, inputs, strength, alpha
        self.layers = model.model.language_model.layers
        # Released LLaVA-32 range is 25..30; keep the same last-seven..last-two
        # offsets for the 36-layer Qwen3B decoder. No tuning on the 500 test IDs.
        self.sla_layers = list(range(max(0, len(self.layers) - 7), max(1, len(self.layers) - 1)))
        self.handles = []

    def _last_states(self, inputs):
        captured = [None] * len(self.layers)
        handles = []
        for index, layer in enumerate(self.layers):
            def capture(module, args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden[0, -1].detach().float().cpu()
            handles.append(layer.register_forward_hook(capture))
        try:
            self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(captured)

    def __enter__(self):
        start, end = vision_range(self.model, self.inputs.input_ids)
        ids = torch.cat([self.inputs.input_ids[:, :start], self.inputs.input_ids[:, end + 1:]], dim=1)
        negative = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
        with torch.inference_mode():
            neg = self._last_states(negative)
            pos = self._last_states(dict(self.inputs))
        directions = vista_direction(neg, pos).to(self.inputs.input_ids.device)
        try:
            for index, layer in enumerate(self.layers):
                def steer(module, args, output, index=index):
                    return vista_steer(output, directions[index], self.strength)
                self.handles.append(layer.mlp.register_forward_hook(steer))

            def augment(module, args, output):
                if output.hidden_states is None:
                    raise ValueError('VISTA SLA requires intermediate hidden states.')
                logits = [module.lm_head(output.hidden_states[index + 1][:, -1:, :]) for index in self.sla_layers]
                augmented = torch.stack(logits).mean(0)
                output.logits = self.alpha * augmented + (1 - self.alpha) * output.logits
                return output
            self.handles.append(self.model.register_forward_hook(augment))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class CausalLensAttention(Qwen2_5_VLAttention):
    def __init__(self, config, layer_idx, strength=0.15):
        super().__init__(config, layer_idx)
        self.strength = strength
        self.sys_len = 0
        self.img_len = 0

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_values=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        batch, queries, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, queries, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_scaling['mrope_section'])
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx,
                {'sin': sin, 'cos': cos, 'cache_position': cache_position})
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        length = k.shape[-2]
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask[:, :, :, :length]
        else:
            # SDPA can omit a materialized causal mask. The explicit adapter
            # must preserve causality for the image/text prefill as well.
            query_positions = cache_position if cache_position is not None else torch.arange(length - queries, length, device=q.device)
            future = torch.arange(length, device=q.device)[None, :] > query_positions[-queries:, None]
            scores = scores.masked_fill(future[None, None, :, :], -float('inf'))
        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        start, end = min(self.sys_len, length), min(self.sys_len + self.img_len, length)
        h_sys = torch.matmul(weights[..., :start], v[..., :start, :])
        h_vis = torch.matmul(weights[..., start:end], v[..., start:end, :])
        h_rest = torch.matmul(weights[..., end:], v[..., end:, :])
        h_lang = h_sys + h_rest
        h_orig = h_lang + h_vis
        visual_weights = weights[..., start:end]
        if end - start < 2:
            raise ValueError('CausalLens needs a nonempty visual token range.')
        sensitivity = visual_weights.var(dim=-1, keepdim=True) / (visual_weights.mean(dim=-1, keepdim=True) + 1e-6)
        sensitivity = sensitivity / (sensitivity.mean(dim=1, keepdim=True) + 1e-6)
        e_vis, e_lang = (h_vis ** 2).mean(), (h_lang ** 2).mean()
        gamma = (e_lang / (e_vis + e_lang + 1e-6)).view(1, 1, 1, 1)
        head = (1 - gamma) * h_orig + gamma * (h_lang + self.strength * sensitivity * (h_vis - h_lang))
        result = self.o_proj(head.transpose(1, 2).contiguous().view(batch, queries, self.hidden_size))
        delta = (h_vis - h_lang).transpose(1, 2).contiguous().view(batch, queries, self.hidden_size)
        result = result + self.strength * sensitivity.mean(dim=1) * self.o_proj(delta)
        return result, weights if output_attentions else None


class CausalLens:
    def __init__(self, model, strength=0.15, layer_start=10, layer_end=20):
        self.model, self.originals, self.adapters = model, [], []
        for index, layer in enumerate(model.model.language_model.layers):
            if layer_start < index < layer_end:
                old = layer.self_attn
                adapter = CausalLensAttention(old.config, index, strength)
                adapter.load_state_dict(old.state_dict())
                adapter.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype).eval()
                layer.self_attn = adapter
                self.originals.append((layer, old))
                self.adapters.append(adapter)

    def update(self, input_ids):
        start, end = vision_range(self.model, input_ids)
        for adapter in self.adapters:
            adapter.sys_len, adapter.img_len = start, end - start

    def restore(self):
        for layer, original in self.originals:
            layer.self_attn = original

