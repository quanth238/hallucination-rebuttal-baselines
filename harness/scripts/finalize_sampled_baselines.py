"""Verify and report the three baselines under the documented repository policy."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math

from experiment_runtime import ROOT, CONTROL, require_execution


METHODS = ('flb', 'vista', 'causallens')
REPORT = 'coco500_sampled_bacnbt2_v1'


def read(path):
    return json.loads(path.read_text())


def main():
    require_execution()
    from calibrate_vista import caption_checks

    manifest_path = CONTROL / 'data/coco500.json'
    manifest = read(manifest_path)
    ids = manifest['image_ids']
    policy_path = CONTROL / 'reports/protocol_audit_bacnbt2_20261108/sampled_protocol.json'
    policy = read(policy_path)
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    assert len(ids) == len(set(ids)) == 500
    results, provenance = [], {}
    for method in METHODS:
        directory = ROOT / 'outputs' / f'{method}_{REPORT}'
        rows = [json.loads(line) for line in (directory / 'captions.jsonl').read_text().splitlines()]
        run, score, details = [read(directory / name) for name in
                               ('run.json', 'scores.json', 'per_image_scores.json')]
        assert run['method'] == method and run['paper_protocol_sha256'] == policy_hash
        assert run['paper_protocol'] == policy
        assert run['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert read(directory / 'completed.json')['count'] == score['count'] == 500
        assert [r['image_id'] for r in rows] == [r['image_id'] for r in details['sentences']] == ids
        assert [r['caption'] for r in rows] == [r['caption'] for r in details['sentences']]
        assert all(r['sample_seed'] == policy['base_seed'] + policy['sample_seed_offset'] + r['image_id']
                   and r['caption'].strip() and 0 < len(r['token_ids']) <= 512
                   and r['generated_tokens'] == len(r['token_ids']) and r['generation_seconds'] > 0
                   for r in rows)
        assert score['scorer_environment']['nltk'] == policy['scorer_nltk_version']
        assert score['scorer_environment']['inflect'] == policy['scorer_inflect_version']
        sentences = details['sentences']
        counts = {'captions': 500,
            'hallucinating_captions': sum(bool(r['mscoco_hallucinated_words']) for r in sentences),
            'hallucinated_object_mentions': sum(len(r['mscoco_hallucinated_words']) for r in sentences),
            'generated_object_mentions': sum(len(r['mscoco_generated_words']) for r in sentences),
            'recalled_ground_truth_objects': sum(len(set(r['mscoco_generated_words']) & set(r['mscoco_gt_words'])) for r in sentences),
            'ground_truth_objects': sum(len(r['mscoco_gt_words']) for r in sentences)}
        assert counts == score['counts'] == details['counts']
        tokens = sum(r['generated_tokens'] for r in rows)
        seconds = sum(r['generation_seconds'] for r in rows)
        metrics = {'CHAIR_S': 100 * counts['hallucinating_captions'] / 500,
            'CHAIR_I': 100 * counts['hallucinated_object_mentions'] / counts['generated_object_mentions'],
            'object_recall': 100 * counts['recalled_ground_truth_objects'] / counts['ground_truth_objects'],
            'average_tokens': tokens / 500, 'cap_hits': sum(r['cap_hit'] for r in rows),
            'observed_tokens_per_second': tokens / seconds,
            'total_generation_seconds': seconds,
            'peak_vram_gb': max(r['peak_allocated_bytes'] for r in rows) / 1024**3}
        assert all(math.isclose(value, score['metrics'][key], rel_tol=1e-10, abs_tol=1e-10)
                   for key, value in metrics.items())
        results.append({'method': method, **metrics, **counts,
                        'repetition_or_language_flags': sum(caption_checks(r['caption'])['degenerate'] for r in rows)})
        provenance[method] = {'run': run, 'scorer_environment': score['scorer_environment'],
            'scorer_implementation_sha256': score['scorer_implementation_sha256'],
            'file_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                           for name in ('captions.jsonl', 'run.json', 'scores.json', 'per_image_scores.json')}}
    report = ROOT / 'reports' / REPORT
    report.mkdir(parents=True, exist_ok=True)
    payload = {'status': 'completed_verified_against_documented_repository_policy',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': policy,
        'protocol_sha256': policy_hash, 'image_ids': ids, 'results': results,
        'provenance': provenance, 'selection_protocol': manifest.get('selection_protocol', {}),
        'paper_row_certification': 'Exact latest POTION runtime remains unverified because its raw configuration/captions are not committed.',
        'excluded_methods': ['potion', 'new_full_native_evaluation']}
    (report / 'comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (report / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    names = {'flb': 'FLB', 'vista': 'VISTA (Qwen adaptation)', 'causallens': 'CausalLens'}
    lines = ['# Baselines under the documented bacnbt_2 policy', '',
        'All three methods cover the exact supplied 500 ordered image IDs. Sampling: T=1, top-p=1, top-k=0, repetition penalty=1; per-image seed=30260817+image_id; 512-token cap. Scoring uses NLTK3.10.3, which exactly reproduces all500 archived Native lexical records and reference scores.', '',
        '| Method | CHAIR-S ↓ | CHAIR-I ↓ | Object recall ↑ | Avg tokens | Cap hits | Tok/s | Flags |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in results:
        lines.append(f"| {names[row['method']]} | {row['CHAIR_S']:.2f} | {row['CHAIR_I']:.2f} | {row['object_recall']:.2f} | {row['average_tokens']:.1f} | {row['cap_hits']} | {row['observed_tokens_per_second']:.2f} | {row['repetition_or_language_flags']} |")
    lines += ['', 'CHAIR/recall are percentages; recall is auxiliary micro object recall.', '',
        'This reconstructs the repository’s documented common policy. The exact newer31.4/5.61 POTION run configuration and captions are not committed; full historical equivalence is not certified. See protocol.evidence/limitations in comparison.json.', '',
        'The cohort was selected post hoc using POT outcomes. FLB/VISTA are Qwen2.5 adaptations; CausalLens is adapted from its released Qwen2 adapter. VISTA retains the previously frozen lambda=.1, alpha=.03, layers29..34 without test-driven retuning. All captions and flagged cases remain included.', '',
        'Timing is measured on RTX4090 and includes image/method preparation plus generation. Keep it separate from H100 paper throughput. No new Native or POTION full run was performed.']
    (report / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': payload['status'], 'report': str(report), 'results': results}), flush=True)


if __name__ == '__main__':
    main()
