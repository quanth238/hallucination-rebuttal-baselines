"""Trace archived CHAIR parsing differences without running model inference."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

from experiment_runtime import ROOT, CONTROL, require_execution


def main():
    require_execution()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    import nltk
    from transformers import AutoTokenizer

    sys.path.insert(0, str(CONTROL))
    from evaluation.official_chair import OfficialChairEvaluator, singularize

    directory = CONTROL / 'reports/protocol_audit_bacnbt2_20261108'
    output = ROOT / 'reports/protocol_audit_bacnbt2_20261108'
    if args.label:
        if not args.label.replace('_', '').isalnum():
            raise ValueError('Use an alphanumeric audit label.')
        output = output / args.label
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in
            (directory / 'reference_native_1024.jsonl').read_text().splitlines()]
    old = json.loads((directory / 'reference_native_full_details.json').read_text())
    truth = {int(k): set(v) for k, v in
             json.loads((ROOT / 'data/coco500/ground_truth.json').read_text())['objects'].items()}
    evaluator = OfficialChairEvaluator(ROOT / 'repos/CHAIR/data/synonyms.txt')
    scored = evaluator.compute(
        [{'image_id': row['image_id'], 'caption': row['response']} for row in rows], truth)
    diffs = []
    for reference, current, row in zip(old['sentences'], scored['sentences'], rows):
        assert reference['image_id'] == current['image_id'] == row['image_id']
        assert reference['caption'] == current['caption']
        changed = {key: {'old': reference[key], 'current': current[key]}
                   for key in ('mscoco_gt_words', 'mscoco_generated_words',
                               'mscoco_hallucinated_words', 'words')
                   if reference[key] != current[key]}
        if changed:
            tokens = nltk.word_tokenize(row['response'].lower())
            diffs.append({'image_id': row['image_id'], 'source_tokens': len(row['token_ids']),
                          'changes': changed,
                          'singularized_tokens': [[word, singularize(word)] for word in tokens]})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen2.5-VL-3B-Instruct',
                                              local_files_only=True)
    prefixes = [{'image_id': row['image_id'], 'caption': row['response']
                 if len(row['token_ids']) <= 512 else tokenizer.decode(
                     row['token_ids'][:512], skip_special_tokens=True,
                     clean_up_tokenization_spaces=False)} for row in rows]
    prefix_details = evaluator.compute(prefixes, truth)
    for filename, data in [('current_native_full_details.json', scored),
                           ('current_native_prefix512_details.json', prefix_details)]:
        (output / filename).write_text(json.dumps(data, ensure_ascii=False) + '\n')
    report = {'status': 'completed', 'scope': 'CPU-only archived caption scoring',
              'versions': {name: importlib.metadata.version(name)
                           for name in ('nltk', 'inflect', 'transformers')},
              'old_counts': old['counts'], 'current_counts': scored['counts'],
              'prefix512_counts': prefix_details['counts'],
              'different_images': len(diffs), 'differences': diffs}
    (output / 'chair_lexical_replay.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'differences'}), flush=True)
    for diff in diffs:
        print(json.dumps({'image_id': diff['image_id'], 'changed_fields': list(diff['changes'])}), flush=True)


if __name__ == '__main__':
    main()
