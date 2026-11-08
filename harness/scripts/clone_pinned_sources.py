"""Prepare the same official source revisions on the selected experiment host."""
import json
from pathlib import Path
import subprocess

from experiment_runtime import ROOT, CONTROL, require_execution


def main():
    require_execution()
    sources = json.loads((CONTROL / 'sources.lock.json').read_text())['repositories']
    for source in sources:
        path = ROOT / 'repos' / source['name']
        if not (path / '.git').exists():
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', '-C', str(path), 'init', '--quiet'], check=True)
            subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', source['url']], check=True)
        current = subprocess.run(['git', '-C', str(path), 'rev-parse', '--verify', 'HEAD'],
                                 text=True, capture_output=True)
        if current.stdout.strip() != source['commit']:
            if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
                raise SystemExit(f'Preserve modified source checkout: {path}')
            subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', source['commit']], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', source['commit']], check=True)
        print(f"Ready: {source['name']} {source['commit']}", flush=True)


if __name__ == '__main__':
    main()
