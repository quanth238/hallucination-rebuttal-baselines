#!/usr/bin/env python3
"""CPU setup: pinned Qwen model, exactly the supplied 500 JPEGs, and CHAIR assets."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

from experiment_runtime import ROOT, CONTROL, backend, execution_id, require_execution


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def download(url, target):
    target = Path(target)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as output:
        shutil.copyfileobj(source, output)
    temporary.replace(target)
    return target


def main():
    require_execution()
    if os.environ.get('REBUTTAL_ROOT') != str(ROOT):
        raise SystemExit('Source the execution environment for this host.')
    from huggingface_hub import snapshot_download
    from PIL import Image
    import nltk

    lock = json.loads((CONTROL / 'sources.lock.json').read_text())
    model = lock['model']
    print('Downloading/reusing pinned Qwen2.5-VL-3B model', flush=True)
    model_link = ROOT / 'models/Qwen2.5-VL-3B-Instruct'
    snapshot = Path(snapshot_download(
        repo_id=model['id'], revision=model['revision'], max_workers=2,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.tiktoken', '*.jinja'],
        local_dir=model_link if backend() == 'vast' else None,
    ))
    if model_link.exists() or model_link.is_symlink():
        if model_link.resolve() != snapshot.resolve():
            raise SystemExit('Existing model link has a different revision.')
    else:
        model_link.symlink_to(snapshot, target_is_directory=True)
    indexes = list(snapshot.glob('*.safetensors.index.json'))
    weight_files = sorted(set(json.loads(indexes[0].read_text())['weight_map'].values())) if indexes else [p.name for p in snapshot.glob('*.safetensors')]
    if not weight_files or any(not (snapshot / name).is_file() for name in weight_files):
        raise SystemExit('Model weight snapshot is incomplete.')

    source = lock['coco500']
    manifest_path = CONTROL / source['manifest']
    if digest(manifest_path) != source['manifest_sha256']:
        raise SystemExit('User-supplied manifest checksum mismatch.')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['rows']
    ids = [int(row['image_id']) for row in rows]
    if len(ids) != 500 or len(set(ids)) != 500 or ids != manifest['image_ids']:
        raise SystemExit('Expected exactly the supplied ordered 500 image IDs.')
    images = ROOT / 'data/coco500/images'
    images.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        name = row['image_file']
        if name != f"COCO_val2014_{int(row['image_id']):012d}.jpg":
            raise ValueError(f'Unexpected COCO filename: {name}')
        target = download(source['image_base_url'] + name, images / name)
        if digest(target) != row['image_sha256']:
            raise ValueError(f'COCO checksum mismatch: {name}')
        with Image.open(target) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError(f'COCO dimensions mismatch: {name}')
            image.verify()

    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(fetch, rows), 1):
            if count % 50 == 0:
                print(f'COCO images verified: {count}/500', flush=True)

    print('Preparing official COCO annotation files and NLTK tokenizer', flush=True)
    archive = download(source['annotations_url'], ROOT / 'cache/coco/annotations_trainval2014.zip')
    annotations = ROOT / 'data/coco500/annotations'
    annotations.mkdir(parents=True, exist_ok=True)
    annotation_hashes = {}
    with zipfile.ZipFile(archive) as zipped:
        for name in ['instances_train2014.json', 'instances_val2014.json', 'captions_train2014.json', 'captions_val2014.json']:
            target = annotations / name
            if not target.exists():
                with zipped.open('annotations/' + name) as source_file, target.open('wb') as output:
                    shutil.copyfileobj(source_file, output)
            annotation_hashes[name] = digest(target)
    for resource in ['punkt', 'punkt_tab']:
        nltk.download(resource, download_dir=os.environ['NLTK_DATA'], raise_on_error=True)
    from transformers import AutoConfig, AutoProcessor
    AutoConfig.from_pretrained(model_link, local_files_only=True)
    AutoProcessor.from_pretrained(model_link, local_files_only=True, use_fast=False)
    marker = {
        'status': 'completed', 'job_id': execution_id(), 'backend': backend(),
        'model': model, 'model_path': str(model_link),
        'weight_files': {name: (snapshot / name).stat().st_size for name in weight_files},
        'coco500': {'images_verified': 500, 'image_ids': ids, 'images_path': str(images),
                    'manifest_sha256': digest(manifest_path), 'annotation_sha256': annotation_hashes},
    }
    (ROOT / 'state/assets.json').write_text(json.dumps(marker, indent=2) + '\n')
    print(json.dumps(marker, indent=2))


if __name__ == '__main__':
    main()

