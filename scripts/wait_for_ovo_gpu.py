#!/usr/bin/env python3
"""Defer the frozen OVO queue, then let its GPU availability checks schedule runs.

This waiter reserves the existing queue lock during the delay, so a second
queue cannot start early. It never stops training processes or changes models.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo


def local_time(timestamp):
    return datetime.fromtimestamp(timestamp, ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def run_waiter(base, command, root, delay_hours, poll_seconds=30):
    if not math.isfinite(delay_hours) or delay_hours < 0:
        raise ValueError('delay_hours must be finite and nonnegative')
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError('poll_seconds must be in (0, 60]')
    base = Path(base)
    status = base/'background_wait_status.json'
    (base/'logs').mkdir(parents=True, exist_ok=True)
    with (base/'background_wait.lock').open('a') as waiter_lock:
        fcntl.flock(waiter_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (base/'schedule.lock').open('a') as reservation:
            fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ready_at = time.time() + delay_hours*3600
            state = dict(phase='delay', pid=os.getpid(), command=command,
                         ready_at_epoch=ready_at, ready_at_china=local_time(ready_at),
                         created_at_china=local_time(time.time()),
                         gpu_policy='After delay: >=28 GiB free and <10% utilization on two polls 30 seconds apart.',
                         sampler_does_not_use_gpu=True)

            def publish():
                state['updated_at_china'] = local_time(time.time())
                temporary = status.with_suffix('.tmp')
                temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False))
                temporary.replace(status)

            publish()
            while time.time() < ready_at:
                state['remaining_delay_seconds'] = max(0, ready_at-time.time())
                publish()
                time.sleep(min(poll_seconds, state['remaining_delay_seconds']))
        # The native scheduler acquires this same lock. A racing manual launch
        # will cause a visible failure, never two simultaneous native queues.
        try:
            with (base/'logs/deferred_queue.log').open('a') as log:
                child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                state.update(phase='queue_running', queue_pid=child.pid,
                             queue_started_epoch=time.time(), remaining_delay_seconds=0)
                publish()
                while child.poll() is None:
                    time.sleep(poll_seconds)
                    publish()
                state.update(phase='queue_completed' if child.returncode == 0 else 'queue_failed',
                             queue_returncode=child.returncode)
                publish()
                return child.returncode
        except Exception as error:
            state.update(phase='waiter_failed', error=repr(error))
            publish()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delay-hours', type=float, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root/'build/benchmarks/ovo_full_20261110'
    assert json.loads((base/'model_artifacts.json').read_text())['complete']
    command = [str(base/'venv311/bin/python'), str(base/'tools/schedule_mapping.py')]
    assert all(Path(p).is_file() for p in command)
    raise SystemExit(run_waiter(base, command, root, args.delay_hours))


if __name__ == '__main__':
    main()
