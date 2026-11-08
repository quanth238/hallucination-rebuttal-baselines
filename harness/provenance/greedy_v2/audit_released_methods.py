"""Compare actual-model greedy execution with the pinned released method kernels."""
import ast
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
from typing import Optional, Tuple

from experiment_runtime import ROOT, CONTROL, require_execution, validate_gpu


def main():
    require_execution()
    import torch
    from torch import nn
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.cache_utils import Cache
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
    from flb import FlbContext
    from qwen_ports import Vista, CausalLens, vision_range, vista_direction

    validate_gpu(torch)
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    path = ROOT / 'models/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', local_files_only=True).eval()
    protocol = json.loads((CONTROL / 'reports/method_audit/vista_calibration_protocol.json').read_text())
    image_id = protocol['image_ids'][0]
    with Image.open(ROOT / 'data/vista_development' / f'COCO_val2014_{image_id:012d}.jpg') as image:
        image = image.convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': protocol['prompt']}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors='pt').to('cuda')
    start, end = vision_range(model, inputs.input_ids)
    report = {'image_id': image_id, 'dtype': 'bfloat16', 'tokens_per_comparison': 32, 'methods': {}}

    def generate(context, hidden=False):
        with torch.inference_mode(), context:
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                num_beams=1, use_cache=True, repetition_penalty=1.0, output_hidden_states=hidden)
        return output[0, inputs.input_ids.shape[1]:].tolist()

    source_path = ROOT / 'repos/FLB/vcd_utils/vcd_sample.py'
    tree = ast.parse(source_path.read_text())
    blocks = [node for node in ast.walk(tree)
              if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
              and isinstance(node.test.values[0], ast.Name) and node.test.values[0].id == 'use_flb']
    assert len(blocks) == 2
    kernel = compile(ast.Module(body=sorted(blocks, key=lambda n: n.lineno), type_ignores=[]), str(source_path), 'exec')

    @contextmanager
    def released_flb():
        namespace = dict(torch=torch, use_flb=True, flb_gamma=0.3, flb_lambda=0.05,
                         model_kwargs={'cd_beta': 0.1}, image_only_logits=None, step=0)
        def hook(module, args, output):
            namespace['next_token_logits'] = output.logits[:, -1, :]
            exec(kernel, namespace)
            output.logits = namespace['next_token_logits'][:, None, :]
            namespace['step'] += 1
            return output
        handle = model.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    port = generate(FlbContext(model))
    released = generate(released_flb())
    report['methods']['flb'] = {'exact_token_match': port == released,
        'port_token_ids': port, 'released_token_ids': released,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest()}
    assert port == released, 'FLB actual-model trajectory differs from released arithmetic'

    sys.path.insert(0, str(ROOT / 'repos/VISTA'))
    llm_layers = importlib.import_module('llm_layers')
    steering_vector = importlib.import_module('steering_vector')
    # The sole change to the released VSV class preserves Qwen's BF16 dtype.
    # Its hard-coded FP16 return otherwise breaks BF16 residual/linear operations.
    source = (ROOT / 'repos/VISTA/llm_layers.py').read_text()
    patched = source.replace('x = x.float()', 'original_dtype = x.dtype\n            x = x.float()').replace('return x.half()', 'return x.to(original_dtype)')
    namespace = {}
    exec(compile(patched, 'VISTA_llm_layers_dtype_compatibility', 'exec'), namespace)
    llm_layers.VSVLayer = namespace['VSVLayer']
    negative_ids = torch.cat([inputs.input_ids[:, :start], inputs.input_ids[:, end + 1:]], dim=1)
    negative = {'input_ids': negative_ids, 'attention_mask': torch.ones_like(negative_ids)}
    with torch.inference_mode():
        direction, _ = steering_vector.obtain_vsv(SimpleNamespace(), model, [[negative, dict(inputs)]], rank=1)
        probe = Vista(model, inputs)
        port_direction = vista_direction(probe._last_states(negative), probe._last_states(dict(inputs)))
    torch.testing.assert_close(port_direction, direction, rtol=0, atol=0)
    sla_path = ROOT / 'repos/VISTA/llava/model/language_model/llava_llama.py'
    sla_tree = ast.parse(sla_path.read_text())
    sla_block = next(n for n in ast.walk(sla_tree) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
                     and n.test.func.id == 'hasattr' and len(n.test.args) > 1
                     and isinstance(n.test.args[1], ast.Constant) and n.test.args[1].value == 'logits_aug')
    sla_kernel = compile(ast.Module(body=[sla_block], type_ignores=[]), str(sla_path), 'exec')

    @contextmanager
    def released_vista():
        llm_layers.add_vsv_layers(model, direction[:, None, :].to('cuda'), [0.1])
        model.logits_aug, model.logits_alpha, model.logits_layers = True, 0.1, '29,34'
        def hook(module, args, output):
            namespace = dict(self=module, outputs=output, logits=output.logits, torch=torch)
            exec(sla_kernel, namespace)
            output.logits = namespace['logits']
            return output
        handle = model.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
            llm_layers.remove_vsv_layers(model)
            steering_vector.remove_logits_flag(model)

    port = generate(Vista(model, inputs, strength=0.1, alpha=0.1), hidden=True)
    released = generate(released_vista(), hidden=True)
    report['methods']['vista'] = {'exact_direction_match': True, 'exact_token_match': port == released,
        'port_token_ids': port, 'released_token_ids': released,
        'compatibility_change': 'Released VSVLayer returns input dtype instead of hard-coded FP16; no arithmetic changes.',
        'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'sla_source_sha256': hashlib.sha256(sla_path.read_bytes()).hexdigest()}
    assert port == released, 'VISTA actual-model trajectory differs from released routines'

    causal_path = ROOT / 'repos/CausalLens/qwen2ours_pope.py'
    tree = ast.parse(causal_path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen2VLAttnAdapter')
    namespace = dict(torch=torch, nn=nn, F=F, math=math, Optional=Optional, Tuple=Tuple, Cache=Cache,
                     repeat_kv=repeat_kv, apply_multimodal_rotary_pos_emb=apply_multimodal_rotary_pos_emb)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(causal_path), 'exec'), namespace)
    original_class = namespace['Qwen2VLAttnAdapter']

    @contextmanager
    def adapted_causal():
        adapters = CausalLens(model)
        adapters.update(inputs.input_ids)
        try:
            yield
        finally:
            adapters.restore()

    @contextmanager
    def released_causal():
        replaced = []
        for index in range(11, 20):
            layer = model.model.language_model.layers[index]
            old = layer.self_attn
            adapter = original_class(old.config, layer_idx=index, lambda_causal=0.15,
                                     sys_len=start, img_len=end - start).to(device='cuda', dtype=torch.bfloat16).eval()
            adapter.load_state_dict(old.state_dict())
            released_forward = adapter.forward
            def forward(this, hidden_states, _released=released_forward, **kwargs):
                # Qwen SDPA omits its causal mask when implicit masking suffices.
                # The released eager adapter must materialize the same mask.
                if kwargs.get('attention_mask') is None:
                    queries = hidden_states.shape[1]
                    positions = kwargs['cache_position']
                    length = int(positions[-1]) + 1
                    future = torch.arange(length, device=hidden_states.device)[None, :] > positions[-queries:, None]
                    mask = torch.zeros((1, 1, queries, length), device=hidden_states.device, dtype=hidden_states.dtype)
                    kwargs['attention_mask'] = mask.masked_fill(future[None, None], -float('inf'))
                return _released(hidden_states, **kwargs)
            adapter.forward = MethodType(forward, adapter)
            layer.self_attn = adapter
            replaced.append((layer, old))
        try:
            yield
        finally:
            for layer, original in replaced:
                layer.self_attn = original

    port = generate(adapted_causal())
    released = generate(released_causal())
    report['methods']['causallens'] = {'exact_token_match': port == released,
        'port_token_ids': port, 'released_token_ids': released,
        'compatibility_change': 'Qwen2.5 RoPE/cache helpers; materialize causal mask when SDPA omits it.',
        'source_sha256': hashlib.sha256(causal_path.read_bytes()).hexdigest()}
    assert port == released, 'CausalLens actual-model trajectory differs from released adapter'
    sys.path.insert(0, str(ROOT / 'repos/POTION'))
    from core_vlm.qwen_extractor import QwenVLExtractor, PreparedInputs
    from core_vlm.custom_generation import QOTSteeredGenerator, deep_update
    from run_coco500 import potion_config
    config = potion_config(32, {'null_penalty': 0.9206608533859253})
    generator = QOTSteeredGenerator(QwenVLExtractor(model, processor), config)
    prepared = PreparedInputs(messages=messages, text=prompt, inputs=dict(inputs), image=image, image_size=image.size)
    from contextlib import nullcontext
    native = generate(nullcontext())
    control_config = deep_update(config, {'steering': {'mode': 'no_steering', 'decode_scope': 'full_vocab'}})
    control = QOTSteeredGenerator(generator.extractor, control_config)
    with torch.inference_mode():
        result = control.generate(prepared=prepared, return_traces=True, detailed_traces=False)
    report['potion_native_control'] = {'exact_token_match': native == result.token_ids,
        'hf_token_ids': native, 'custom_loop_token_ids': result.token_ids,
        'same_prepared_input_tensors': True}
    assert native == result.token_ids, 'POTION unsteered custom loop differs from native greedy generation'
    with torch.inference_mode():
        result = generator.generate(prepared=prepared, return_traces=True, detailed_traces=False)
    projections = [trace for trace in result.traces if trace.kl_after is not None]
    max_kl = max((trace.kl_after for trace in projections), default=0)
    report['potion'] = {'token_ids': result.token_ids, 'projected_steps': len(projections),
                        'max_kl_after': max_kl, 'frozen_configuration': config}
    assert projections and max_kl <= 0.10001, 'POTION correction/KL check failed'
    report['status'] = 'all_actual_model_source_parity_checks_passed'
    output = ROOT / 'reports/method_audit/actual_model_source_parity.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
