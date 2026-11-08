"""Verify the single transferred batch without downloading duplicate assets."""
import hashlib
import json
from pathlib import Path
import subprocess

from experiment_runtime import ROOT, CONTROL, require_execution


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    require_execution()
    lock = json.loads((CONTROL / 'sources.lock.json').read_text())
    manifest_path = CONTROL / lock['coco500']['manifest']
    assert digest(manifest_path) == lock['coco500']['manifest_sha256'], 'Manifest checksum mismatch'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['rows']) == 500
    assert [row['image_id'] for row in manifest['rows']] == manifest['image_ids']
    from PIL import Image
    for row in manifest['rows']:
        path = ROOT / 'data/coco500/images' / row['image_file']
        assert digest(path) == row['image_sha256'], f'Image checksum mismatch: {path}'
        with Image.open(path) as image:
            assert image.size == (row['width'], row['height'])
            image.verify()
    repositories = {}
    for source in lock['repositories']:
        commit = subprocess.check_output(['git', '-C', str(ROOT / 'repos' / source['name']),
            'rev-parse', 'HEAD'], text=True).strip()
        assert commit == source['commit'], f"Revision mismatch: {source['name']}"
        repositories[source['name']] = commit
    model = ROOT / 'models/Qwen2.5-VL-3B-Instruct'
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    weights = {name: (model / name).stat().st_size for name in set(index['weight_map'].values())}
    assert sum(weights.values()) == 7509337976, 'Incomplete transferred weights'
    expected_weights = json.loads((CONTROL / 'data/model_files.sha256.json').read_text())
    model_hashes = {}
    for name, expected in expected_weights.items():
        assert weights[name] == expected['bytes'], f'Model size mismatch: {name}'
        model_hashes[name] = digest(model / name)
        assert model_hashes[name] == expected['sha256'], f'Model checksum mismatch: {name}'
    from transformers import AutoConfig, AutoProcessor
    AutoConfig.from_pretrained(model, local_files_only=True)
    AutoProcessor.from_pretrained(model, local_files_only=True, use_fast=False)
    marker = {'status': 'completed', 'backend': 'vast', 'model': lock['model'],
              'model_path': str(model), 'weight_files': weights,
              'weight_sha256': model_hashes,
              'repositories': repositories, 'images_verified': 500,
              'manifest_sha256': digest(manifest_path)}
    (ROOT / 'state/assets.json').write_text(json.dumps(marker, indent=2) + '\n')
    print(json.dumps(marker, indent=2))


if __name__ == '__main__':
    main()
