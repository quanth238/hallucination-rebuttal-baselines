"""Run FLB -> VISTA -> CausalLens once, then score/report; no monitoring loop."""
from datetime import datetime, timezone
import json
import subprocess

from experiment_runtime import ROOT, CONTROL, require_execution


def main():
    require_execution()
    import nltk

    if nltk.__version__ != '3.10.3':
        raise SystemExit('Use the isolated repository-matched NLTK3.10.3 overlay.')
    audit = ROOT / 'reports/protocol_audit_bacnbt2_20261108'
    if json.loads((audit / 'sampled_source_parity.json').read_text())['status'] != 'all_sampled_baseline_source_parity_checks_passed':
        raise SystemExit('Complete sampled released-kernel comparisons first.')
    lexical = json.loads((audit / 'nltk3103/chair_lexical_replay.json').read_text())
    if lexical['different_images'] != 0 or lexical['old_counts'] != lexical['current_counts']:
        raise SystemExit('Archived CHAIR parsing must reproduce exactly before full scoring.')
    policy = CONTROL / 'reports/protocol_audit_bacnbt2_20261108/sampled_protocol.json'
    frozen = ROOT / 'reports/method_audit/vista_frozen_v2.json'
    python = str(ROOT / 'envs/qwen/bin/python')
    stages = []
    for method in ('flb', 'vista', 'causallens'):
        output = ROOT / 'outputs' / f'{method}_coco500_sampled_bacnbt2_v1'
        arguments = ['run_coco500.py', '--method', method, '--paper-protocol', str(policy),
                     '--output', str(output)]
        if method == 'vista':
            arguments += ['--vista-config', str(frozen)]
        stages += [(f'evaluate-{method}', arguments),
                   (f'score-{method}', ['score_coco500.py', '--captions', str(output / 'captions.jsonl')])]
    stages.append(('report', ['finalize_sampled_baselines.py']))
    state_path = ROOT / 'state/sampled_baselines.json'
    for name, arguments in stages:
        command = [python, str(CONTROL / 'scripts' / arguments[0]), *arguments[1:]]
        log_path = ROOT / 'logs' / f'sampled-{name}.log'
        state = {'status': 'running', 'stage': name, 'scope': 'three_baselines_only',
            'excluded_methods': ['potion', 'new_full_native_evaluation'],
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
    state_path.write_text(json.dumps({'status': 'completed', 'scope': 'three_baselines_only',
        'report': str(ROOT / 'reports/coco500_sampled_bacnbt2_v1/comparison.json'),
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
