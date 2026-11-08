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
METHODS = ('flb', 'vista', 'causallens')
NAMES = {'regular': 'Native control', 'flb': 'FLB', 'vista': 'VISTA (Qwen adaptation)', 'causallens': 'CausalLens'}
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
    lines = ['# Qwen hallucination baseline tracking', '',
        'FLB, VISTA and CausalLens source snapshots, the Qwen2.5 adapters actually used, and run artifacts for the ICLR rebuttal comparison.', '',
        f"**Snapshot prepared:** {time:%Y-%m-%d %H:%M} Vietnam time (UTC+7). This is updated on request; it is not a live monitor.", '',
        '## Current sampled runs', '',
        'The three methods run sequentially: **FLB → VISTA → CausalLens**, using the same supplied 500 images. These target the documented `bacnbt_2` sampling policy. Exact historical output reproduction is not required.', '',
        '| Method | Status | Images | CHAIR-S ↓ | CHAIR-I ↓ | Object recall ↑ | Avg tokens |',
        '|---|---|---:|---:|---:|---:|---:|']
    stage = (snapshot.get('pipeline') or {}).get('stage', '')
    for method in METHODS:
        record = snapshot['sampled'][method]
        metrics = record['metrics']
        state = '🟢 Complete' if metrics else ('🟠 Scoring' if record['completed'] else
                ('🟠 Running' if stage == 'evaluate-' + method or record['count'] else '⚪ Queued'))
        values = ' | '.join(f'{metrics[key]:.2f}' for key in
                            ('CHAIR_S', 'CHAIR_I', 'object_recall', 'average_tokens')) if metrics else '— | — | — | —'
        lines.append(f"| {NAMES[method]} | {state} | {record['count']}/500 | {values} |")
    lines += ['', 'Scores are published after a full 500-image method finishes and passes scoring. Partial caption files are retained as progress snapshots. Only the three requested baselines are scheduled; no new POTION or full Native run is part of this pipeline.', '',
        '## Completed greedy experiment — supplementary', '',
        '**These rows use greedy decoding and legacy NLTK3.9.2 scoring. Keep this table separate from the sampled paper comparison.** The Native row is an existing control.', '',
        '| Method | Images | CHAIR-S ↓ | CHAIR-I ↓ | Object recall ↑ | Avg tokens | Tok/s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in GREEDY:
        record = snapshot['greedy'][method]
        metrics = record['metrics']
        if metrics:
            values = ' | '.join(f'{metrics[key]:.2f}' for key in
                               ('CHAIR_S', 'CHAIR_I', 'object_recall', 'average_tokens', 'observed_tokens_per_second'))
            lines.append(f"| {NAMES[method]} | {record['count']} | {values} |")
    lines += ['', 'CHAIR and recall are percentages. Recall is auxiliary micro object recall. Tokens/s was measured on an RTX4090 and must stay separate from H100 paper throughput.', '',
        'In the greedy experiment, VISTA had 19 automated repetition/language flags and 102 cap hits; all captions remain included. FLB retained 309 hallucinated mentions versus Native309; VISTA had345 and CausalLens218. Read recall, lengths and raw counts alongside the hallucination ratios.', '',
        '## Shared sampled settings', '',
        '- `Qwen/Qwen2.5-VL-3B-Instruct`, revision `66285546d2b821cf421d4f5eb2576359d3770cd3`; BF16, batch1, KV cache.',
        '- Exact ordered 500 COCO val2014 IDs and image hashes in [`harness/data/coco500.json`](harness/data/coco500.json).',
        '- Prompt: `Please describe this image in detail.`',
        '- Sampling: temperature1, top-p1, generation top-k0, repetition penalty1, one beam, 512 new tokens.',
        '- Per-image Torch seed: `30260817 + image_id`, reset before image preparation.',
        '- Fast image processor; `max_pixels=1003520`.',
        '- Sampled scoring uses an isolated NLTK3.10.3 overlay. This exactly reproduces all500 saved archived Native lexical records and reference metrics.',
        '- Frozen method parameters: FLB gamma=.3/rate=.05/beta=.1; VISTA lambda=.1/alpha=.03/layers29–34; CausalLens lambda=.15/layers11–19.', '',
        'The supplied cohort was selected post hoc using POT outcomes; it is not an independent random COCO benchmark. FLB/VISTA are source-checked Qwen adaptations, and CausalLens is adapted from its released Qwen2 adapter. The newer POTION31.4/5.61 row appears in the paper repository report, but its raw run configuration is not committed. Remaining historical defaults are disclosed rather than chased through more reruns.', '',
        'Exact settings: [`sampled_protocol.json`](harness/reports/protocol_audit_bacnbt2_20261108/sampled_protocol.json).', '',
        '## Source snapshots and execution code', '',
        '| Method | Code in this repository | Pinned upstream commit |',
        '|---|---|---|']
    for source in read(REPO / 'methods/SOURCES.json'):
        name = source['name']
        url = source['url'].removesuffix('.git')
        lines.append(f"| {name} | [`methods/{name}/`](methods/{name}/) | [`{source['commit'][:12]}`]({url}/tree/{source['commit']}) |")
    lines += ['', 'Upstream source files are unchanged. Generated Python cache files are omitted; licenses and attribution are retained in [`NOTICE.md`](NOTICE.md). The actual Qwen modifications are in [`harness/scripts/flb.py`](harness/scripts/flb.py) and [`harness/scripts/qwen_ports.py`](harness/scripts/qwen_ports.py). The exact executed runner and controller are also in `harness/scripts/`; old source versions remain under `harness/provenance/`.', '',
        'Both greedy and sampled actual-model 32-token comparisons against released arithmetic passed. Sampled full-cap execution pilots passed for all three methods. These checks establish implementation consistency, not perfect grounding or caption quality.', '',
        '## Saved artifacts', '',
        '- [`results/greedy/`](results/greedy/): completed captions, token IDs, run metadata, scores and per-image object matches.',
        '- [`results/sampled/`](results/sampled/): current sampled captions/metadata; completed metrics appear as the pipeline finishes.',
        '- [`tracking/snapshot.json`](tracking/snapshot.json): counts, metrics, status and artifact hashes used to build this README.',
        '- [`harness/`](harness/): source locks, adapters, runners, scorer, 500-image manifest and frozen configurations.',
        '- [`docs/`](docs/): setup/layout notes and the recorded environment.', '',
        '## Refresh tracking', '',
        'On the original Mac, run one bounded status/sync and push:', '',
        '```bash', 'python3 tracking/update_results.py --refresh --push', '```', '',
        'The local `.tracking.local.json` supplies the original harness/backup paths and is excluded from Git. On another machine, point at an existing backup:', '',
        '```bash', 'python3 tracking/update_results.py --source /path/to/baseline-backup --status /path/to/status.json', '```', '',
        'The refresh command copies results and rebuilds this README. It does not launch experiments or install monitoring. Model weights, COCO images, environments, credentials and SSH connection settings are excluded.']
    text = '\n'.join(lines) + '\n'
    for old, new in [('Native309', 'Native 309'), ('had345', 'had 345'), ('CausalLens218', 'CausalLens 218'),
                     ('batch1', 'batch 1'), ('temperature1', 'temperature 1'), ('top-p1', 'top-p 1'),
                     ('top-k0', 'top-k 0'), ('penalty1', 'penalty 1'), ('NLTK3.', 'NLTK 3.'),
                     ('all500', 'all 500'), ('POTION31.4', 'POTION 31.4')]:
        text = text.replace(old, new)
    return text


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
        'greedy': {}, 'sampled': {}}
    for method, folder in GREEDY.items():
        snapshot['greedy'][method] = copy_outputs(source / 'outputs' / folder, REPO / 'results/greedy' / method, ids)
    for method in METHODS:
        snapshot['sampled'][method] = copy_outputs(source / 'outputs' / f'{method}_coco500_sampled_bacnbt2_v1', REPO / 'results/sampled' / method, ids)
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
