#!/usr/bin/env python3
"""Score the complete supplied 500-image cohort with the paper's CHAIR utility."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from experiment_runtime import ROOT, CONTROL, backend, execution_id, require_execution
sys.path.insert(0, str(CONTROL))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--captions', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require_execution()
    from evaluation.official_chair import OfficialChairEvaluator, annotation_provenance, official_source_provenance

    manifest_path = CONTROL / 'data/coco500.json'
    manifest = json.loads(manifest_path.read_text())
    ids = manifest['image_ids']
    if len(ids) != 500 or len(set(ids)) != 500:
        raise SystemExit('Expected exactly the supplied 500 IDs.')
    chair_root = ROOT / 'repos/CHAIR'
    annotations = ROOT / 'data/coco500/annotations'
    source_hashes = official_source_provenance(chair_root)
    annotation_hashes = annotation_provenance(annotations)
    signature = {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 'source_sha256': source_hashes, 'annotation_sha256': annotation_hashes}
    evaluator = OfficialChairEvaluator(chair_root / 'data/synonyms.txt')
    cache = ROOT / 'data/coco500/ground_truth.json'
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored['signature'] != signature:
            raise SystemExit('Ground-truth cache provenance differs from this run.')
        truth = {int(k): set(v) for k, v in stored['objects'].items()}
    else:
        truth = evaluator.build_ground_truth(annotations, ids)
        cache.write_text(json.dumps({'signature': signature,
            'objects': {str(k): sorted(v) for k, v in truth.items()}}, indent=2) + '\n')
    if set(truth) != set(ids):
        raise SystemExit('Ground-truth cache does not match the selected 500 images.')
    if args.prepare_only:
        marker = {'status': 'completed', 'images': 500, 'job_id': execution_id(), 'backend': backend(),
                  'cache': str(cache), 'signature': signature}
        (ROOT / 'state/scorer.json').write_text(json.dumps(marker, indent=2) + '\n')
        print(json.dumps(marker, indent=2))
        return
    if args.captions is None:
        raise SystemExit('Supply --captions or --prepare-only.')
    rows = [json.loads(line) for line in args.captions.read_text().splitlines() if line.strip()]
    if [row['image_id'] for row in rows] != ids:
        raise SystemExit('Final reporting requires all 500 ordered IDs, with no omissions or extras.')
    details = evaluator.compute(rows, truth)
    metrics = {'CHAIR_S': 100 * details['overall_metrics']['CHAIRs'],
               'CHAIR_I': 100 * details['overall_metrics']['CHAIRi'],
               'object_recall': 100 * details['auxiliary_metrics']['object_recall'],
               'average_tokens': sum(row['generated_tokens'] for row in rows) / 500,
               'cap_hits': sum(row['cap_hit'] for row in rows),
               'total_generation_seconds': sum(row['generation_seconds'] for row in rows),
               'peak_vram_gb': max(row['peak_allocated_bytes'] for row in rows) / 1024**3,
               'observed_tokens_per_second': sum(row['generated_tokens'] for row in rows) / sum(row['generation_seconds'] for row in rows)}
    destination = args.output or args.captions.parent / 'scores.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {'count': 500, 'metrics': metrics, 'signature': signature,
               'scorer_environment': {key: importlib.metadata.version(key) for key in ('nltk', 'inflect')},
               'scorer_implementation_sha256': hashlib.sha256((CONTROL / 'evaluation/official_chair.py').read_bytes()).hexdigest(),
               'selection_protocol': manifest.get('selection_protocol', {}),
               'recall_provenance': 'auxiliary micro object recall',
               'timing_scope': 'method preparation and GPU generation; excludes queue/model loading',
               'counts': details['counts']}
    destination.write_text(json.dumps(payload, indent=2) + '\n')
    destination.with_name('per_image_scores.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()

