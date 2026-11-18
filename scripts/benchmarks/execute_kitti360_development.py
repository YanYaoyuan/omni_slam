"""Run only the fixed development-window image frontend, one capped GPU job."""
import os
from pathlib import Path
import subprocess
import time
from paper_final_common import ROOT, read, dump, sha, PY
from prepare_kitti360_frontend import OUT
from execute_hypersim_shared_gpu import samples


def main():
    job_path = OUT/'jobs/2013_05_28_drive_0000_sync.json'
    job = read(job_path)
    assert job['kind'] == 'kitti360_development' and len(job['frames']) == 60
    status_path = OUT/'frontend_execution.json'
    assert not status_path.exists(), 'Inspect an existing execution before attempting recovery'
    state = dict(pid=os.getpid(), complete=False, failed=False, stage='starting', child_pid=None,
        job_sha256=sha(job_path), script_sha256=sha(__file__), completed=[], started_epoch=time.time())
    def publish():
        state['updated_epoch'] = time.time()
        dump(status_path, state)
    publish()
    logs = OUT/'logs'
    logs.mkdir(exist_ok=True)
    old = ROOT/'build/benchmarks/ovo_full_20261110'
    env = dict(os.environ, HF_HOME=str(old/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
        MPLCONFIGDIR='/tmp/omni-kitti-mpl', PYTHONPATH=str(ROOT/'omni_semantic_map'))
    try:
        for phase in job['phases']:
            assert phase in ['masks', 'triplets', 'features']
            assert sha(job_path) == state['job_sha256']
            last = None
            while True:
                rows = samples()
                state.update(stage='wait_capacity', gpu_samples=rows)
                publish()
                good = [r for r in rows if '5090' in r['name'] and r['free_mib'] >= 10240]
                chosen = max(good, key=lambda r:r['free_mib'])['index'] if good else None
                if chosen is not None and chosen == last: break
                last = chosen
                time.sleep(10)
            log_path = logs/(phase+'.log')
            with log_path.open('x') as log:
                process = subprocess.Popen([str(PY), 'scripts/benchmarks/split_strong_frontend.py',
                    '--job', str(job_path), '--phase', phase], cwd=ROOT,
                    env=dict(env, CUDA_VISIBLE_DEVICES=str(chosen)), stdout=log, stderr=subprocess.STDOUT)
                state.update(stage=phase, child_pid=process.pid, gpu=chosen, log=str(log_path))
                publish()
                while process.poll() is None:
                    publish()
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError(f'{phase} exited {process.returncode}; inspect logs before retry')
            run = read(Path(job['output'])/(phase+'_run.json'))
            assert run['complete'] and run['job_sha256'] == state['job_sha256']
            for name, expected in run['files_sha256'].items():
                assert sha(Path(job['output'])/name) == expected
            state['completed'].append(phase)
            state['child_pid'] = None
            publish()
            print('DONE', phase, flush=True)
        state.update(complete=True, stage='complete', child_pid=None, finished_epoch=time.time())
        publish()
    except Exception as error:
        state.update(failed=True, error=repr(error), child_pid=None)
        publish()
        raise


if __name__ == '__main__': main()
