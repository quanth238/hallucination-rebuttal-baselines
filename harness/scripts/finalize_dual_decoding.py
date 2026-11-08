"""Verify the four sampled rows and compare them with the saved greedy captions."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math

from experiment_runtime import ROOT, CONTROL, require_execution

METHODS = ('regular', 'flb', 'vista', 'causallens')
NAMES = {'regular': 'Regular', 'flb': 'FLB', 'vista': 'VISTA', 'causallens': 'CausalLens'}
METRICS = ('CHAIR_S', 'CHAIR_I', 'object_recall', 'average_tokens', 'observed_tokens_per_second')


def read(path):
    return json.loads(path.read_text())


def main():
    require_execution()
    manifest = read(CONTROL / 'data/coco500.json')
    ids = manifest['image_ids']
    greedy = read(ROOT / 'reports/greedy_protocol_check_20261108/comparison.json')
    greedy_rows = {row['method']: row for row in greedy['results']}
    sampled, provenance = [], {}
    settings = ('model', 'manifest_sha256', 'dtype', 'do_sample', 'temperature', 'top_p', 'top_k',
                'repetition_penalty', 'max_new_tokens', 'num_beams', 'processor_use_fast',
                'max_pixels', 'base_seed', 'sample_seed_offset', 'attn_implementation')
    reference_run = None
    for method in METHODS:
        folder = ROOT / 'outputs' / f'{method}_coco500_sampled_bacnbt2_v1'
        run, score, details = [read(folder / name) for name in ('run.json', 'scores.json', 'per_image_scores.json')]
        rows = [json.loads(line) for line in (folder / 'captions.jsonl').read_text().splitlines()]
        assert run['method'] == method and run['do_sample'] is True
        assert run['temperature'] == run['top_p'] == run['repetition_penalty'] == 1.0 and run['top_k'] == 0
        assert read(folder / 'completed.json')['count'] == score['count'] == 500
        assert [row['image_id'] for row in rows] == ids == [row['image_id'] for row in details['sentences']]
        assert [row['caption'] for row in rows] == [row['caption'] for row in details['sentences']]
        assert all(row['sample_seed'] == 30260817 + row['image_id'] and
                   0 < len(row['token_ids']) == row['generated_tokens'] <= 512 for row in rows)
        assert score['scorer_environment']['nltk'] == '3.10.3'
        assert details['counts'] == score['counts']
        assert run['manifest_sha256'] == hashlib.sha256((CONTROL / 'data/coco500.json').read_bytes()).hexdigest()
        if reference_run is not None:
            assert all(run[key] == reference_run[key] for key in settings), method
        reference_run = run
        sentences = details['sentences']
        hallucinating = sum(bool(row['mscoco_hallucinated_words']) for row in sentences)
        wrong = sum(len(row['mscoco_hallucinated_words']) for row in sentences)
        mentions = sum(len(row['mscoco_generated_words']) for row in sentences)
        recalled = sum(len(set(row['mscoco_generated_words']) & set(row['mscoco_gt_words'])) for row in sentences)
        truth = sum(len(row['mscoco_gt_words']) for row in sentences)
        recalculated = {'CHAIR_S': 100 * hallucinating / 500, 'CHAIR_I': 100 * wrong / mentions,
                        'object_recall': 100 * recalled / truth}
        assert all(math.isclose(value, score['metrics'][key], rel_tol=1e-10) for key, value in recalculated.items())
        sampled.append({'method': method, **score['metrics'], **score['counts']})
        provenance[method] = {'run': run, 'scorer_environment': score['scorer_environment'],
            'artifact_sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                               for name in ('captions.jsonl', 'run.json', 'scores.json', 'per_image_scores.json')}}
    deltas = [{'method': row['method'], **{key: row[key] - greedy_rows[row['method']]['metrics'][key]
               for key in METRICS}} for row in sampled]
    output = ROOT / 'reports/dual_decoding_coco500'
    output.mkdir(parents=True, exist_ok=True)
    payload = {'status': 'completed_verified_dual_decoding_comparison',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'count_per_method_per_protocol': 500,
        'greedy': greedy, 'sampled': sampled, 'sampled_minus_greedy': deltas,
        'sampled_provenance': provenance,
        'limitations': ['Greedy uses the saved slow-processor runs; sampling uses the recorded fast-processor policy. This is a comparison of the two recorded protocols, not a strict decoder-only ablation.',
            'Greedy timing excludes image preparation; sampled timing includes it. Keep timing scopes explicit.',
            'The supplied500-image cohort was selected post hoc using POT outcomes.',
            'These new measurements are separate from historic paper rows; no POTION generation is included.']}
    (output / 'comparison.json').write_text(json.dumps(payload, indent=2) + '\n')
    with (output / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['protocol', 'method', *METRICS])
        writer.writeheader()
        for method in METHODS:
            writer.writerow({'protocol': 'greedy', 'method': method,
                             **{key: greedy_rows[method]['metrics'][key] for key in METRICS}})
        for row in sampled:
            writer.writerow({'protocol': 'sampling_T1', 'method': row['method'], **{key: row[key] for key in METRICS}})
    lines = ['# Greedy and T=1 sampling', '', 'Qwen2.5-VL-3B, the same500 images and prompt;512-token cap; NLTK3.10.3 scoring.', '']
    for title, group in [('Greedy', [{'method': m, **greedy_rows[m]['metrics']} for m in METHODS]), ('Sampling T=1', sampled)]:
        lines += [f'## {title}', '', '| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |', '|---|---:|---:|---:|---:|---:|']
        for row in group:
            lines.append('| ' + NAMES[row['method']] + ' | ' + ' | '.join(f'{row[key]:.2f}' for key in METRICS) + ' |')
        lines.append('')
    lines += ['Recorded preprocessing and timing scopes differ between the two protocols; see comparison.json. All captions and source/run hashes are preserved.']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': payload['status'], 'report': str(output)}), flush=True)


if __name__ == '__main__':
    main()
