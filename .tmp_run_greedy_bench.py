import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

instances = [
    'challenge_r200d15_1',
    'challenge_r300d20_1',
    'challenge_r300d20_2',
    'challenge_r300d20_3',
    'challenge_r300d20_4',
    'challenge_r300d20_5',
    'challenge_r500d25_1',
    'challenge_r500d25_2',
    'challenge_r500d25_3',
]
root = Path('.')
bench_file = root / 'logs' / 'runtime_benchmarks.jsonl'
records = []
for inst in instances:
    cmd = [sys.executable, 'main.py', str(root / 'instances' / f'{inst}.txt'), '--method', 'greedy_gls']
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    dt = time.perf_counter() - t0
    tail = '\n'.join((out.stdout or '').splitlines()[-5:])
    rec = {
        'timestamp_utc': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'instance': inst,
        'method': 'greedy_gls',
        'repeat': 1,
        'wall_time_sec': dt,
        'returncode': out.returncode,
        'timed_out': False,
        'run_label': 'method_comparison',
        'stdout_tail': tail,
    }
    records.append(rec)
    print(f"{inst}: {dt:.3f}s rc={out.returncode}")

with open(bench_file, 'a', encoding='utf-8') as fh:
    for rec in records:
        fh.write(json.dumps(rec) + '\n')

print(f"Appended {len(records)} greedy_gls benchmark records to {bench_file}")
