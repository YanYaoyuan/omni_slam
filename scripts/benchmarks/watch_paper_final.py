"""After the live coordinator finishes, audit and render the fixed experiment."""
import os
from pathlib import Path
import subprocess
import time
from paper_final_common import ROOT,RUN,PY,read,dump,sha


def main():
    pid=int((RUN/'coordinator.pid').read_text());proc=Path(f'/proc/{pid}')
    assert b'paper_final_queue.py' in (proc/'cmdline').read_bytes()
    start=(proc/'stat').read_text().split()[21]
    status=dict(complete=False,pid=os.getpid(),coordinator_pid=pid,coordinator_start_ticks=start,stage='waiting_for_coordinator')
    def publish():status['updated_epoch']=time.time();dump(RUN/'delivery_status.json',status)
    publish()
    while True:
        state=read(RUN/'queue_status.json')
        if state['complete']:break
        if state.get('failed'):
            status.update(failed=True,reason='Coordinator failed; inspect its log.');publish();return
        try:
            stat=(proc/'stat').read_text().split();assert stat[21]==start and stat[2]!='Z'
        except (FileNotFoundError,AssertionError):
            status.update(failed=True,reason='Coordinator exited without a complete state.');publish();return
        status['coordinator_stage']=state['stage'];publish();time.sleep(10)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/omni-paper-final-report')
    for step,script in [('audit','audit_paper_final.py'),('report','report_paper_final.py')]:
        log=RUN/'logs'/(step+'_delivery.log');assert not log.exists()
        status.update(stage=step);publish()
        with log.open('x') as f:
            child=subprocess.Popen([str(PY),str(ROOT/'scripts/benchmarks'/script)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;publish();rc=child.wait()
        if rc:
            status.update(failed=True,returncode=rc,log=str(log));publish();return
    status.update(complete=True,stage='complete',child_pid=None,completion_audit_sha256=sha(RUN/'completion_audit.json'))
    publish()


if __name__=='__main__':main()
