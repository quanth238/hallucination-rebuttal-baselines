#!/usr/bin/env python3
"""Copy a baseline backup and rebuild the tracking README; optional one-time push."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
METHODS = ('regular', 'flb', 'vista', 'causallens')
NAMES = {'regular': 'Regular', 'flb': 'FLB', 'vista': 'VISTA', 'causallens': 'CausalLens'}
GREEDY = {'regular': 'regular_coco500_512', 'flb': 'flb_coco500_512',
          'vista': 'vista_coco500_512_v2', 'causallens': 'causallens_coco500_512'}
ARTIFACTS = ('captions.jsonl', 'run.json', 'scores.json', 'per_image_scores.json', 'completed.json')


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def copy_outputs(source, destination, manifest_ids):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    path = destination / 'captions.jsonl'
    lines = path.read_text().splitlines() if path.exists() else []
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or (source / 'completed.json').exists():
                raise
            # A running worker may have been copied during its last append.
            path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    assert [row['image_id'] for row in rows] == manifest_ids[:len(rows)]
    completion = read(destination / 'completed.json')
    if completion:
        assert completion['count'] == len(rows) == 500
    score = read(destination / 'scores.json')
    if score:
        assert score['count'] == len(rows) == 500
    return {'count': len(rows), 'completed': bool(completion),
            'metrics': score['metrics'] if score else None,
            'hashes': {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                       for name in ARTIFACTS if (destination / name).exists()}}


def render(snapshot):
    time = datetime.fromisoformat(snapshot['prepared_at_utc']).astimezone(ZoneInfo('Asia/Ho_Chi_Minh'))
    lines = ['# Hallucination baselines', '',
        'Regular, FLB, VISTA and CausalLens with Qwen2.5-VL-3B-Instruct on the same 500 COCO images.', '',
        f"Results snapshot: {time:%Y-%m-%d %H:%M} (UTC+7).", '',
        'Both protocols use a 512-token cap and NLTK 3.10.3 scoring.', '',
        '## Greedy', '', '`do_sample=False`, 500 images per method.', '',
        '| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |',
        '|---|---:|---:|---:|---:|---:|']
    keys = ('CHAIR_S', 'CHAIR_I', 'object_recall', 'average_tokens', 'observed_tokens_per_second')
    for method in METHODS:
        record = (snapshot.get('greedy_verified') or snapshot['greedy'])[method]
        metrics = record['metrics']
        if metrics:
            values = ' | '.join(f'{metrics[key]:.2f}' for key in keys)
            lines.append(f"| {NAMES[method]} | {values} |")
    lines += ['', '## Sampling T=1', '', '`do_sample=True`, temperature1, top-p1, top-k0;500 images per method.']
    completed_sampled = [(method, snapshot['sampled'][method]['metrics']) for method in METHODS
                         if snapshot['sampled'][method]['metrics']]
    if completed_sampled:
        lines += ['',
            '| Method | CHAIR-S ↓ | CHAIR-I ↓ | Recall ↑ | Avg tokens | Tok/s |',
            '|---|---:|---:|---:|---:|---:|']
        for method, metrics in completed_sampled:
            values = ' | '.join(f'{metrics[key]:.2f}' for key in keys)
            lines.append(f"| {NAMES[method]} | {values} |")
    else:
        lines += ['', 'Completed scores will be added as the sampled runs finish. [Run files](results/sampled/) · [Progress snapshot](tracking/snapshot.json).']
    if (REPO / 'results/sampled/regular/reference_check.json').exists():
        lines += ['', 'Regular T=1 differs from the recorded manuscript baseline; see the [reference comparison](results/sampled/regular/reference_check.json).']
    lines += ['', 'CHAIR and recall are percentages. Timing uses an RTX4090. Recorded preprocessing and timing scopes are described in the [evaluation notes](docs/EVALUATION.md).', '',
        '## Code', '',
        '| Method | Source | Upstream |',
        '|---|---|---|']
    for source in read(REPO / 'methods/SOURCES.json'):
        name = source['name']
        url = source['url'].removesuffix('.git')
        lines.append(f"| {name} | [methods/{name}](methods/{name}/) | [Original repository]({url}/tree/{source['commit']}) |")
    lines += ['', '[Run files and scores](results/) · [Experiment code](harness/) · [Evaluation notes](docs/EVALUATION.md) · [Setup](docs/RUNNING.md)', '',
        '[Verified greedy scores](results/greedy_verified/) retain the original [captions and run records](results/greedy/). Archived repository reference values remain in [results/reference/](results/reference/).', '',
        'The supplied 500-image subset was selected using POT outcomes; selection details are retained in the [image manifest](harness/data/coco500.json).', '',
        '## Update results', '',
        '```bash', 'python3 tracking/update_results.py --refresh --push', '```', '',
        'Upstream source versions and licenses are retained in [SOURCES.json](methods/SOURCES.json) and [NOTICE.md](NOTICE.md).']
    return ('\n'.join(lines) + '\n').replace('temperature1', 'temperature 1').replace('top-p1', 'top-p 1').replace('top-k0;500', 'top-k 0; 500')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--status', type=Path)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--push', action='store_true')
    args = parser.parse_args()
    config = read(REPO / '.tracking.local.json') or {}
    if args.source is None and 'source_dir' not in config:
        parser.error('Provide --source BACKUP_DIR or an ignored .tracking.local.json with source_dir.')
    source = args.source or Path(config['source_dir'])
    status_path = args.status or (Path(config['status_file']) if config.get('status_file') else None)
    if args.refresh:
        if 'harness_dir' not in config:
            parser.error('--refresh requires harness_dir in the ignored local configuration.')
        control = Path(config['harness_dir'])
        status = subprocess.check_output([sys.executable, str(control / 'scripts/vast_control.py'), 'status'], text=True)
        cache = REPO / '.tracking-cache'
        cache.mkdir(exist_ok=True)
        status_path = cache / 'status.json'
        status_path.write_text(status)
        subprocess.run([sys.executable, str(control / 'scripts/vast_control.py'), 'sync-results'], check=True)
    assert source.is_dir(), source
    status = read(status_path) if status_path else {}
    pipeline = (status or {}).get('pipeline') or read(source / 'state/sampled_baselines.json') or {}
    ids = read(REPO / 'harness/data/coco500.json')['image_ids']
    snapshot = {'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'remote_checked_at_utc': (status or {}).get('checked_at_utc'),
        'pipeline': {key: pipeline[key] for key in ('status', 'stage', 'scope') if key in pipeline},
        'greedy': {}, 'greedy_verified': {}, 'sampled': {}}
    for method, folder in GREEDY.items():
        snapshot['greedy'][method] = copy_outputs(source / 'outputs' / folder, REPO / 'results/greedy' / method, ids)
    for method in METHODS:
        snapshot['sampled'][method] = copy_outputs(source / 'outputs' / f'{method}_coco500_sampled_bacnbt2_v1', REPO / 'results/sampled' / method, ids)
    rescored = source / 'reports/greedy_protocol_check_20261108'
    proof = read(rescored / 'comparison.json')
    if proof:
        target = REPO / 'results/greedy_verified'
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rescored / 'comparison.json', target / 'comparison.json')
        for row in proof['results']:
            method = row['method']
            assert snapshot['greedy'][method]['hashes']['captions.jsonl'] == row['original_artifact_sha256']['captions.jsonl']
            assert row['counts']['captions'] == 500 and row['scorer_environment']['nltk'] == '3.10.3'
            destination = target / method
            destination.mkdir(exist_ok=True)
            for name in ('scores.json', 'per_image_scores.json'):
                shutil.copy2(rescored / method / name, destination / name)
            snapshot['greedy_verified'][method] = {'count': 500, 'metrics': row['metrics'],
                'counts': row['counts'], 'scorer_environment': row['scorer_environment'],
                'caption_source': f'results/greedy/{method}/captions.jsonl'}
    dual = source / 'reports/dual_decoding_coco500'
    if (dual / 'comparison.json').exists():
        destination = REPO / 'results/dual_decoding'
        destination.mkdir(exist_ok=True)
        for name in ('comparison.json', 'comparison.csv', 'README.md'):
            shutil.copy2(dual / name, destination / name)
    regular_check = source / 'reports/regular_t1_reference_check.json'
    if regular_check.exists():
        shutil.copy2(regular_check, REPO / 'results/sampled/regular/reference_check.json')
    for variant, folder in [('greedy', 'coco500_matched_v2'), ('sampled', 'coco500_sampled_bacnbt2_v1')]:
        for name in ('comparison.csv', 'comparison.json', 'paper_comparability.json'):
            path = source / 'reports' / folder / name
            if path.exists():
                shutil.copy2(path, REPO / 'results' / variant / name)
    (REPO / 'tracking/snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (REPO / 'README.md').write_text(render(snapshot))
    print(json.dumps({'greedy': {m: v['count'] for m, v in snapshot['greedy'].items()},
                      'sampled': {m: v['count'] for m, v in snapshot['sampled'].items()}}))
    if args.push:
        subprocess.run(['git', 'add', 'README.md', 'results', 'tracking/snapshot.json'], cwd=REPO, check=True)
        diff = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO)
        if diff.returncode == 1:
            subprocess.run(['git', 'commit', '-m', 'Update baseline results and tracking snapshot'], cwd=REPO, check=True)
        elif diff.returncode:
            raise SystemExit('Could not inspect staged tracking updates.')
        subprocess.run(['git', 'push'], cwd=REPO, check=True)


if __name__ == '__main__':
    main()
