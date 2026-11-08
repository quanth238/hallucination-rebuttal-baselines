"""Resume existing T1 baselines, add Regular T1, then compare with saved greedy runs."""
from datetime import datetime, timezone
import hashlib
import json
import subprocess

from experiment_runtime import ROOT, CONTROL, require_execution


def main():
    require_execution()
    original = CONTROL / 'scripts/run_coco500.py'
    if hashlib.sha256(original.read_bytes()).hexdigest() != 'a6c5d3540820e05d421409825e3b5daeb678e6a52a06d4e77b73e1aeebea0894':
        raise SystemExit('Preserve the original sampled checkpoint runner.')
    greedy = json.loads((ROOT / 'reports/greedy_protocol_check_20261108/comparison.json').read_text())
    if greedy['status'] != 'completed_cpu_rescore' or len(greedy['results']) != 4:
        raise SystemExit('The completed greedy comparison must be available first.')
    policy = CONTROL / 'reports/protocol_audit_bacnbt2_20261108/sampled_protocol_with_regular.json'
    regular = ROOT / 'outputs/regular_coco500_sampled_bacnbt2_v1'
    stages = [
        ('sampled-baselines', ['run_sampled_baselines.py']),
        ('evaluate-regular', ['run_regular_sampled.py', '--method', 'regular',
            '--paper-protocol', str(policy), '--output', str(regular)]),
        ('score-regular', ['score_coco500.py', '--captions', str(regular / 'captions.jsonl')]),
        ('report-dual-decoding', ['finalize_dual_decoding.py']),
    ]
    state_path = ROOT / 'state/dual_decoding.json'
    python = str(ROOT / 'envs/qwen/bin/python')
    for name, arguments in stages:
        command = [python, str(CONTROL / 'scripts' / arguments[0]), *arguments[1:]]
        log_path = ROOT / 'logs' / f'dual-{name}.log'
        state = {'status': 'running', 'stage': name, 'scope': 'greedy_and_sampled_regular_plus_three_baselines',
            'excluded_methods': ['potion'], 'greedy_count_per_method': 500,
            'command': command, 'log': str(log_path),
            'updated_at_utc': datetime.now(timezone.utc).isoformat()}
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        with log_path.open('a') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if child.returncode:
            state.update(status='failed', returncode=child.returncode)
            state_path.write_text(json.dumps(state, indent=2) + '\n')
            raise SystemExit(f'{name} failed; inspect {log_path}')
    state_path.write_text(json.dumps({'status': 'completed', 'scope': 'greedy_and_sampled_regular_plus_three_baselines',
        'report': str(ROOT / 'reports/dual_decoding_coco500/comparison.json'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
