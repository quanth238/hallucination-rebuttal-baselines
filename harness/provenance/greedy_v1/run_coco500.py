#!/usr/bin/env python3
"""Generate resumable captions for exactly the user's fixed COCO-500 cohort."""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from experiment_runtime import ROOT, CONTROL, backend, execution_id, require_execution, validate_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['regular', 'flb', 'vista', 'causallens'], required=True)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--image-ids', help='Comma-separated IDs for a diagnostic pilot; full reporting still requires all 500.')
    args = parser.parse_args()
    require_execution()
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
    rows = manifest['rows'][:args.limit]
    if args.image_ids:
        requested = [int(value) for value in args.image_ids.split(',')]
        if len(set(requested)) != len(requested) or not set(requested).issubset(manifest['image_ids']):
            raise SystemExit('Pilot IDs must be unique members of the supplied cohort.')
        rows = [row for row in manifest['rows'] if row['image_id'] in set(requested)]
    output = args.output or ROOT / 'outputs' / f'{args.method}_coco500_{args.max_new_tokens}'
    output.mkdir(parents=True, exist_ok=True)
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
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', local_files_only=True,
    ).eval()
    causal = CausalLens(model) if args.method == 'causallens' else None
    for row in rows[len(done):]:
        image_path = ROOT / 'data/coco500/images' / row['image_file']
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row['image_sha256']:
            raise SystemExit(f'Image checksum mismatch: {image_path}')
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['prompt']}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
        if causal:
            causal.update(inputs.input_ids)
        processors = LogitsProcessorList()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        context = (FlbContext(model) if args.method == 'flb' else
                   Vista(model, inputs) if args.method == 'vista' else nullcontext())
        with torch.inference_mode(), context:
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                use_cache=True, repetition_penalty=1.0, logits_processor=processors,
                output_hidden_states=args.method == 'vista',
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        token_ids = generated[0, inputs.input_ids.shape[1]:].tolist()
        record = {'image_id': row['image_id'], 'caption': processor.decode(token_ids, skip_special_tokens=True),
                  'token_ids': token_ids, 'generated_tokens': len(token_ids),
                  'generation_seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'cap_hit': len(token_ids) == args.max_new_tokens}
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

