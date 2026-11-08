"""Verify and report the three complete methods selected after the VISTA review."""
from datetime import datetime, timezone
import csv
import hashlib
import json
import math

from experiment_runtime import ROOT, CONTROL, require_execution


METHODS = ['regular', 'flb', 'causallens']
MATCHED_FIELDS = ['model', 'manifest_sha256', 'max_new_tokens', 'dtype', 'do_sample',
    'num_beams', 'repetition_penalty', 'attn_implementation', 'processor_use_fast',
    'backend', 'device', 'device_memory_bytes', 'seed', 'ports_sha256', 'flb_sha256',
    'runner_sha256', 'runtime_sha256']


def read(path):
    return json.loads(path.read_text())


def main():
    require_execution()
    manifest_path = CONTROL / 'data/coco500.json'
    manifest = read(manifest_path)
    ids = manifest['image_ids']
    assert len(ids) == len(set(ids)) == 500
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    runs, scores, details, hashes = {}, {}, {}, {}
    results = []
    for method in METHODS:
        folder = ROOT / 'outputs' / f'{method}_coco500_512'
        captions_path = folder / 'captions.jsonl'
        captions = [json.loads(line) for line in captions_path.read_text().splitlines() if line.strip()]
        run, score, per_image = read(folder / 'run.json'), read(folder / 'scores.json'), read(folder / 'per_image_scores.json')
        assert run['method'] == method, f'Mislabeled method metadata: {method}'
        sentences = per_image['sentences']
        assert [row['image_id'] for row in captions] == ids, f'Incomplete or unordered {method} captions'
        assert [row['image_id'] for row in sentences] == ids
        assert all(row['caption'] == item['caption'] for row, item in zip(captions, sentences))
        assert all(row['caption'].strip() and 0 < row['generated_tokens'] <= 512 and
                   row['generated_tokens'] == len(row['token_ids']) and row['generation_seconds'] > 0 for row in captions)
        assert run['manifest_sha256'] == score['signature']['manifest_sha256'] == manifest_hash
        assert score['count'] == 500 and run['purpose'] == 'full_500_image_evaluation'
        counts = {
            'captions': 500,
            'hallucinating_captions': sum(bool(row['mscoco_hallucinated_words']) for row in sentences),
            'hallucinated_object_mentions': sum(len(row['mscoco_hallucinated_words']) for row in sentences),
            'generated_object_mentions': sum(len(row['mscoco_generated_words']) for row in sentences),
            'recalled_ground_truth_objects': sum(len(set(row['mscoco_generated_words']) & set(row['mscoco_gt_words'])) for row in sentences),
            'ground_truth_objects': sum(len(row['mscoco_gt_words']) for row in sentences),
        }
        assert counts == score['counts'] == per_image['counts']
        metrics = {
            'CHAIR_S': 100 * counts['hallucinating_captions'] / 500,
            'CHAIR_I': 100 * counts['hallucinated_object_mentions'] / counts['generated_object_mentions'],
            'object_recall': 100 * counts['recalled_ground_truth_objects'] / counts['ground_truth_objects'],
            'average_tokens': sum(row['generated_tokens'] for row in captions) / 500,
            'cap_hits': sum(row['cap_hit'] for row in captions),
            'total_generation_seconds': sum(row['generation_seconds'] for row in captions),
            'peak_vram_gb': max(row['peak_allocated_bytes'] for row in captions) / 1024**3,
            'observed_tokens_per_second': sum(row['generated_tokens'] for row in captions) / sum(row['generation_seconds'] for row in captions),
        }
        for name, value in metrics.items():
            assert math.isclose(value, score['metrics'][name], rel_tol=1e-10, abs_tol=1e-10), (method, name)
        runs[method], scores[method], details[method] = run, score, per_image
        hashes[method] = hashlib.sha256(captions_path.read_bytes()).hexdigest()
        results.append({'method': method, **metrics, **counts,
                        'generation_minutes': metrics['total_generation_seconds'] / 60})
    reference = runs['regular']
    for method in METHODS[1:]:
        assert all(runs[method][key] == reference[key] for key in MATCHED_FIELDS), f'Unmatched configuration for {method}'
        assert scores[method]['signature'] == scores['regular']['signature']
    paired, differences = {}, {}
    for method in METHODS[1:]:
        counts = {'both_clean': 0, 'fixed_native_hallucination': 0,
                  'added_hallucination': 0, 'both_hallucinate': 0}
        for base, candidate in zip(details['regular']['sentences'], details[method]['sentences']):
            a, b = bool(base['mscoco_hallucinated_words']), bool(candidate['mscoco_hallucinated_words'])
            key = 'both_hallucinate' if a and b else 'fixed_native_hallucination' if a else 'added_hallucination' if b else 'both_clean'
            counts[key] += 1
        paired[method] = counts
        differences[method] = {name + '_change_pp': scores[method]['metrics'][name] - scores['regular']['metrics'][name]
                               for name in ['CHAIR_S', 'CHAIR_I', 'object_recall']}
        differences[method]['hallucinated_object_mentions_change'] = scores[method]['counts']['hallucinated_object_mentions'] - scores['regular']['counts']['hallucinated_object_mentions']
    validity = read(ROOT / 'state/method-validity.json')
    report = ROOT / 'reports/coco500'
    report.mkdir(parents=True, exist_ok=True)
    payload = {'status': 'completed_valid_comparison', 'count_per_method': 500,
        'model': reference['model'], 'matched_settings': {key: reference[key] for key in MATCHED_FIELDS},
        'image_ids': ids, 'selection_protocol': manifest.get('selection_protocol', {}),
        'results': results, 'changes_from_native': differences, 'paired_caption_outcomes': paired,
        'caption_sha256': hashes, 'excluded_methods': {'vista': validity.get('vista', {})},
        'verification': 'Exact ordered IDs, caption consistency, matched settings and all metrics independently checked.',
        'timing_scope': scores['regular']['timing_scope'], 'recall_provenance': scores['regular']['recall_provenance'],
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    (report / 'valid_comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (report / 'valid_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (report / 'README.md').write_text(
        '# Verified Qwen COCO-500 comparison\n\n'
        'Native decoding, FLB and CausalLens each use exactly the same 500 ordered image IDs, pinned '
        'Qwen2.5-VL-3B weights, BF16, greedy decoding, prompt, image processor and 512-token cap. '
        'All results were checked against their per-image captions and scores.\n\n'
        'VISTA is excluded at the user\'s explicit request: its Qwen adaptation produced severe '
        'repetition in the pilot. No VSV-only or tuned variant is substituted as VISTA.\n\n'
        'The supplied cohort was selected post hoc using POT outcomes; it is not an independent '
        'random benchmark sample. Results describe this selected cohort.\n\n'
        'CHAIR-S and CHAIR-I are percentages; lower is better. Object recall is auxiliary micro '
        'recall; higher is better. Generation minutes include per-image method preparation and '
        'GPU generation, excluding model loading and setup. Inspect raw counts and coverage as '
        'well as ratios. FLB has 309 hallucinated object mentions, the same as native, so its '
        'lower CHAIR-I comes from a larger mention denominator.\n')
    print(json.dumps({'results': results, 'changes_from_native': differences,
                      'report': str(report / 'valid_comparison.csv')}, indent=2))


if __name__ == '__main__':
    main()
