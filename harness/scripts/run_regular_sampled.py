#!/usr/bin/env python3
"""Generate the user-requested Regular T=1 control; preserves the original runner logic."""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiment_runtime import ROOT, CONTROL, backend, execution_id, require_execution, validate_gpu


def potion_config(max_new_tokens, parameters):
    """Frozen manuscript configuration, shared by validation and full evaluation."""
    return {
        'extraction': {'cache_image_embeddings': True, 'use_kv_cache': True,
            'lazy_candidate_states': True, 'drop_prefill_inputs_embeds': True,
            'cache_visual_costs': True, 'layer': -1, 'top_k': 5,
            'max_candidates': 5, 'normalize_states': True,
            'filter_object_claims': True, 'accept_open_vocab': False},
        'qot': {'cost_method': 'lm_head_embedding_cosine', 'candidate_wise': False,
            'visual_weight_mode': 'uniform_sum', 'claim_weight_mode': 'uniform_sum',
            'transport_solver': 'pot', 'null_penalty': parameters['null_penalty'], 'tol': 1e-5},
        'steering': {'mode': 'legacy_grounding_risk_kl', 'decode_scope': 'candidate_set',
            'eta': 0.5, 'lambda_penalty': 8.0, 'delta': 1e-6,
            'normalize_grounded_states': False, 'kl_budget': 0.1},
        'generation': {'max_new_tokens': max_new_tokens, 'do_sample': False,
            'temperature': 1.0, 'top_p': 1.0, 'top_k': 0,
            'repetition_penalty': 1.0, 'stop_on_eos': True},
        'logging': {'keep_step_traces': True},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['regular'], required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--image-ids', help='Comma-separated IDs for a diagnostic pilot; full reporting still requires all 500.')
    parser.add_argument('--vista-config', type=Path, help='Frozen development-selected VISTA configuration.')
    parser.add_argument('--paper-protocol', type=Path, help='Explicit documented bacnbt_2 sampling protocol; baseline methods only.')
    args = parser.parse_args()
    require_execution()
    policy = json.loads(args.paper_protocol.read_text()) if args.paper_protocol else None
    if policy is None:
        raise SystemExit('The Regular sampled control requires --paper-protocol.')
    if policy:
        if args.method != 'regular':
            raise SystemExit('This separate runner generates only the Regular sampled control.')
        if policy['name'] != 'bacnbt2_documented_qwen_sampling_v1':
            raise SystemExit('Unknown paper comparison protocol.')
        if args.max_new_tokens != policy['max_new_tokens']:
            raise SystemExit('Use the common documented 512-token quality cap.')
    if not 1 <= args.limit <= 500:
        raise SystemExit('The requested cohort contains exactly 500 images.')
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessorList
    from flb import FlbContext
    from qwen_ports import Vista, CausalLens

    device = validate_gpu(torch)
    torch.set_num_threads(int(os.environ.get('REBUTTAL_THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.manual_seed(20261107)
    manifest_file = CONTROL / 'data/coco500.json'
    manifest = json.loads(manifest_file.read_text())
    if len(manifest['rows']) != 500 or [r['image_id'] for r in manifest['rows']] != manifest['image_ids']:
        raise SystemExit('Manifest IDs differ from the selected cohort.')
    if policy and hashlib.sha256(manifest_file.read_bytes()).hexdigest() != policy['manifest_sha256']:
        raise SystemExit('The paper protocol requires the exact supplied manifest.')
    rows = manifest['rows'][:args.limit]
    if args.image_ids:
        requested = [int(value) for value in args.image_ids.split(',')]
        if len(set(requested)) != len(requested) or not set(requested).issubset(manifest['image_ids']):
            raise SystemExit('Pilot IDs must be unique members of the supplied cohort.')
        rows = [row for row in manifest['rows'] if row['image_id'] in set(requested)]
    if policy and args.output is None:
        raise SystemExit('Use a new explicit output folder; preserve the completed greedy rows.')
    output = args.output or ROOT / 'outputs' / f'{args.method}_coco500_{args.max_new_tokens}'
    output.mkdir(parents=True, exist_ok=True)
    vista_parameters = {'vsv_lambda': 0.17, 'sla_alpha': 0.3, 'sla_layers': list(range(29, 35))}
    if args.vista_config:
        vista_parameters = json.loads(args.vista_config.read_text())['selected']
        if not vista_parameters or not vista_parameters.get('manually_reviewed'):
            raise SystemExit('VISTA requires a stable, manually reviewed development configuration.')
    if args.method == 'vista' and len(rows) == 500 and not args.vista_config:
        raise SystemExit('Full VISTA evaluation requires --vista-config after development validation.')
    metadata = {
        'method': args.method, 'model': json.loads((CONTROL / 'sources.lock.json').read_text())['model'],
        'manifest_sha256': hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        'image_ids': [r['image_id'] for r in rows], 'max_new_tokens': args.max_new_tokens,
        'dtype': 'bfloat16', 'do_sample': False, 'num_beams': 1, 'repetition_penalty': 1.0,
        'attn_implementation': 'sdpa', 'torch': torch.__version__,
        'processor_use_fast': False,
        'backend': backend(), 'device': device.name, 'device_memory_bytes': device.total_memory,
        'seed': 20261107,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'runtime_sha256': hashlib.sha256((CONTROL / 'scripts/experiment_runtime.py').read_bytes()).hexdigest(),
        'flb_correction_stage': 'raw_model_logits_before_standard_generation_processors',
        'ports_sha256': hashlib.sha256((CONTROL / 'scripts/qwen_ports.py').read_bytes()).hexdigest(),
        'flb_sha256': hashlib.sha256((CONTROL / 'scripts/flb.py').read_bytes()).hexdigest(),
        'parameters': {'flb': {'gamma': 0.3, 'rate': 0.05, 'beta': 0.1},
                       'vista': {'vsv_lambda': 0.17, 'sla_alpha': 0.3, 'sla_layer_offsets': [-7, -2]},
                       'causallens': {'lambda_causal': 0.15, 'layer_start_exclusive': 10, 'layer_end_exclusive': 20}},
        'purpose': 'full_500_image_evaluation' if [r['image_id'] for r in rows] == manifest['image_ids'] and args.max_new_tokens == 512 else 'execution_or_pilot',
    }
    if args.method == 'vista':
        metadata['parameters']['vista'] = vista_parameters
        metadata['vista_config_sha256'] = hashlib.sha256(args.vista_config.read_bytes()).hexdigest() if args.vista_config else None
        metadata['implementation_status'] = 'Qwen2.5 port retaining released VSV + SLA, with disjoint architecture stability calibration'
    if policy:
        if metadata['model'] != {'id': policy['model_id'], 'revision': policy['model_revision']}:
            raise SystemExit('Pinned model differs from the reconstructed protocol.')
        if any(row['prompt'] != policy['prompt'] for row in rows):
            raise SystemExit('Prompt differs from the paper protocol.')
        metadata.update({key: policy[key] for key in ('do_sample', 'temperature', 'top_p', 'top_k',
            'repetition_penalty', 'presence_penalty', 'processor_use_fast', 'max_pixels',
            'base_seed', 'sample_seed_offset', 'seed_policy')})
        metadata['seed'] = None
        metadata['paper_protocol'] = policy
        metadata['paper_protocol_sha256'] = hashlib.sha256(args.paper_protocol.read_bytes()).hexdigest()
        metadata['timing_scope'] = 'image preparation, method preparation and GPU generation; excludes model loading and image checksum validation'
    if args.method == 'potion':
        snapshot = json.loads((CONTROL / 'reports/method_audit/potion_source_snapshot.json').read_text())
        for name, expected_hash in snapshot['files'].items():
            path = ROOT / 'repos/POTION' / name
            # The executable snapshot contains Python modules; configuration is
            # recorded in source provenance and resolved explicitly below.
            if name.endswith('.py') and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise SystemExit(f'POTION source checksum mismatch: {name}')
        metadata['potion_source_snapshot'] = snapshot
        metadata['parameters']['potion'] = snapshot['manuscript_parameters']
        metadata['implementation_status'] = 'Frozen original paper implementation, linear POT with full grounding/risk KL projection'
    run_file = output / 'run.json'
    if run_file.exists() and json.loads(run_file.read_text()) != metadata:
        raise SystemExit('Existing run uses different settings; use a new output directory.')
    run_file.write_text(json.dumps(metadata, indent=2) + '\n')
    captions_file = output / 'captions.jsonl'
    done = [json.loads(line) for line in captions_file.read_text().splitlines() if line.strip()] if captions_file.exists() else []
    if [r['image_id'] for r in done] != [r['image_id'] for r in rows[:len(done)]]:
        raise SystemExit('Partial results do not match the expected ordered image IDs.')
    if len(done) == len(rows):
        print(f'Already complete: {captions_file}')
        return
    model_path = ROOT / 'models/Qwen2.5-VL-3B-Instruct'
    processor_kwargs = {'use_fast': policy['processor_use_fast'], 'max_pixels': policy['max_pixels']} if policy else {'use_fast': False}
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    potion = None
    if args.method == 'potion':
        sys.path.insert(0, str(ROOT / 'repos/POTION'))
        from core_vlm.qwen_extractor import QwenVLExtractor
        from core_vlm.custom_generation import QOTSteeredGenerator
        potion_parameters = metadata['parameters']['potion']
        config = potion_config(args.max_new_tokens, potion_parameters)
        (output / 'resolved_potion_config.json').write_text(json.dumps(config, indent=2) + '\n')
        potion = QOTSteeredGenerator(QwenVLExtractor(model, processor), config)
    causal = CausalLens(model) if args.method == 'causallens' else None
    for row in rows[len(done):]:
        image_path = ROOT / 'data/coco500/images' / row['image_file']
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['image_sha256']:
            raise SystemExit(f'Image checksum mismatch: {image_path}')
        if policy:
            torch.manual_seed(policy['base_seed'] + policy['sample_seed_offset'] + row['image_id'])
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
        if causal:
            causal.update(inputs.input_ids)
        processors = LogitsProcessorList()
        if not policy:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
        context = FlbContext(model) if args.method == 'flb' else nullcontext()
        if args.method == 'vista':
            context = Vista(model, inputs, strength=vista_parameters['vsv_lambda'], alpha=vista_parameters['sla_alpha'])
            context.sla_layers = vista_parameters['sla_layers']
        with torch.inference_mode(), context:
            if potion:
                from core_vlm.qwen_extractor import PreparedInputs
                prepared = PreparedInputs(messages=messages, text=text, inputs=dict(inputs), image=image, image_size=image.size)
                generated = potion.generate(prepared=prepared, max_new_tokens=args.max_new_tokens,
                    return_traces=True, detailed_traces=False)
            else:
                generation = {'do_sample': False, 'num_beams': 1, 'use_cache': True, 'repetition_penalty': 1.0}
                if policy:
                    generation.update({key: policy[key] for key in ('do_sample', 'temperature', 'top_p',
                        'top_k', 'num_beams', 'use_cache', 'repetition_penalty')})
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, **generation, logits_processor=processors,
                    output_hidden_states=args.method == 'vista',
                )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        token_ids = generated.token_ids if potion else generated[0, inputs.input_ids.shape[1]:].tolist()
        record = {'image_id': row['image_id'], 'caption': processor.decode(token_ids, skip_special_tokens=True),
                  'token_ids': token_ids, 'generated_tokens': len(token_ids),
                  'generation_seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'cap_hit': len(token_ids) == args.max_new_tokens}
        if policy:
            record['caption'] = processor.tokenizer.decode(token_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False).strip()
            record['sample_seed'] = policy['base_seed'] + policy['sample_seed_offset'] + row['image_id']
            record['processed_visual_grid'] = inputs.image_grid_thw[0].tolist()
        if potion:
            interventions = [trace for trace in generated.traces if trace.kl_after is not None]
            record['potion_diagnostics'] = {
                'projected_steps': len(interventions),
                'max_kl_after': max((trace.kl_after for trace in interventions), default=0),
                'mean_kl_after': sum(trace.kl_after for trace in interventions) / max(1, len(interventions)),
                'mean_projection_scale': sum(trace.kl_scale for trace in interventions) / max(1, len(interventions)),
                'transport_solver': 'linear_partial_pot',
            }
            if record['potion_diagnostics']['max_kl_after'] > 0.10001:
                raise SystemExit('POTION KL constraint violated.')
        if not token_ids:
            raise SystemExit(f"Empty output at image {row['image_id']}")
        with captions_file.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        done.append(record)
        print(json.dumps({'method': args.method, 'completed': len(done), 'requested': len(rows),
                          'image_id': row['image_id'], 'tokens': len(token_ids), 'seconds': elapsed}), flush=True)
        del generated, inputs
    if causal:
        causal.restore()
    (output / 'completed.json').write_text(json.dumps({'status': 'completed', 'count': len(done),
        'job_id': execution_id(), 'backend': backend(), 'device': device.name, 'device_memory_bytes': device.total_memory}) + '\n')


if __name__ == '__main__':
    main()

