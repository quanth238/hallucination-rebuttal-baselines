"""Frozen, development-only VISTA stability calibration on the actual Qwen model."""
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import urlopen

from experiment_runtime import ROOT, CONTROL, require_execution, validate_gpu


def caption_checks(text):
    words = re.findall(r"\w+", text.lower())
    grams = Counter(tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4)))
    repeated = sum(n - 1 for n in grams.values()) / max(1, sum(grams.values()))
    cjk = sum('\u3400' <= c <= '\u9fff' for c in text) / max(1, len(text))
    maximum = max(grams.values(), default=0)
    return {'max_5gram_count': maximum, 'repeated_5gram_fraction': repeated,
            'cjk_character_fraction': cjk,
            'degenerate': maximum >= 5 or repeated >= 0.15 or cjk > 0.05}


def summarize(protocol, saved, signature):
    native = [r for r in saved if r['variant'] == 'regular']
    assert len(native) == len(protocol['image_ids'])
    native_mean = sum(r['generated_tokens'] for r in native) / len(native)
    native_caps = sum(r['cap_hit'] for r in native)
    summary = {'protocol_sha256': signature, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
               'development_test_overlap': 0, 'count': len(saved), 'variants': [], 'selected': None}
    higher_priority_complete = True
    for variant in protocol['variants']:
        rows = [r for r in saved if r['variant'] == variant['name']]
        mean = sum(r['generated_tokens'] for r in rows) / max(1, len(rows))
        caps = sum(r['cap_hit'] for r in rows)
        degenerate = sum(r['degenerate'] for r in rows)
        complete = len(rows) == len(protocol['image_ids'])
        cap_ok = caps <= native_caps + 2 if protocol.get('cap_gate', True) else True
        stable = complete and degenerate == 0 and mean >= 0.5 * native_mean and cap_ok
        summary['variants'].append({**variant, 'count': len(rows), 'complete': complete,
            'degenerate': degenerate, 'mean_tokens': mean, 'cap_hits': caps, 'stable': stable})
        if stable and higher_priority_complete and summary['selected'] is None:
            summary['selected'] = {**variant, 'sla_layers': protocol['sla_layers']}
        higher_priority_complete = higher_priority_complete and complete
    if summary['selected'] and any(not row['complete'] for row in summary['variants']):
        summary['early_stop_rationale'] = ('The predeclared rule selects the first stable variant. '
            'Every higher-priority variant was completed and failed. Lower-priority runs cannot change selection and are unnecessary.')
    destination = ROOT / 'reports/method_audit' / (protocol.get('output_name', 'vista_calibration_v2') + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if summary['selected'] is None:
        raise SystemExit('No stable VISTA configuration: full evaluation remains blocked.')


def main():
    require_execution()
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_ports import Vista

    validate_gpu(torch)
    torch.set_num_threads(4)
    torch.manual_seed(20261107)
    protocol_path = Path(os.environ.get('VISTA_PROTOCOL', CONTROL / 'reports/method_audit/vista_calibration_protocol.json'))
    protocol = json.loads(protocol_path.read_text())
    test = json.loads((CONTROL / 'data/coco500.json').read_text())
    assert not set(protocol['image_ids']) & set(test['image_ids'])
    output = ROOT / 'outputs' / protocol.get('output_name', 'vista_calibration_v2')
    output.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    registry = output / 'protocol.json'
    if registry.exists() and registry.read_bytes() != protocol_path.read_bytes():
        raise ValueError('Calibration protocol changed; use a new version.')
    registry.write_bytes(protocol_path.read_bytes())
    results = output / 'captions.jsonl'
    saved = [json.loads(line) for line in results.read_text().splitlines()] if results.exists() else []
    if '--summarize-only' in sys.argv:
        summarize(protocol, saved, signature)
        return
    image_dir = ROOT / 'data/vista_development'
    image_dir.mkdir(parents=True, exist_ok=True)
    assets = {}
    for image_id in protocol['image_ids']:
        path = image_dir / f'COCO_val2014_{image_id:012d}.jpg'
        if not path.exists():
            with urlopen(f'https://s3.amazonaws.com/images.cocodataset.org/val2014/{path.name}', timeout=45) as response:
                data = response.read()
            with Image.open(__import__('io').BytesIO(data)) as image:
                image.verify()
            path.write_bytes(data)
        assets[str(image_id)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / 'assets.json').write_text(json.dumps(assets, indent=2) + '\n')
    model_path = ROOT / 'models/Qwen2.5-VL-3B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa', local_files_only=True).eval()
    variants = [{'name': 'regular'}] + protocol['variants']
    expected = [(v['name'], i) for v in variants for i in protocol['image_ids']]
    assert [(r['variant'], r['image_id']) for r in saved] == expected[:len(saved)]
    for variant in variants:
        for image_id in protocol['image_ids']:
            if (variant['name'], image_id) in {(r['variant'], r['image_id']) for r in saved}:
                continue
            with Image.open(image_dir / f'COCO_val2014_{image_id:012d}.jpg') as image:
                image = image.convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': protocol['prompt']}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to('cuda')
            context = nullcontext()
            if variant['name'] != 'regular':
                context = Vista(model, inputs, strength=variant['vsv_lambda'], alpha=variant['sla_alpha'])
                context.sla_layers = protocol['sla_layers']
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode(), context:
                generated = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                    num_beams=1, use_cache=True, repetition_penalty=1.0,
                    output_hidden_states=variant['name'] != 'regular')
            torch.cuda.synchronize()
            tokens = generated[0, inputs.input_ids.shape[1]:].tolist()
            caption = processor.decode(tokens, skip_special_tokens=True)
            row = {'variant': variant['name'], 'image_id': image_id,
                   'caption': caption, 'token_ids': tokens, 'generated_tokens': len(tokens),
                   'cap_hit': len(tokens) == 512, 'generation_seconds': time.monotonic() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), **caption_checks(caption)}
            with results.open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            saved.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in {'caption', 'token_ids'}}), flush=True)
            del generated, inputs
    summarize(protocol, saved, signature)


if __name__ == '__main__':
    main()
