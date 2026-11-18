"""Wait for one verified converter process, then start the full validation once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit_evidence_retention import ROOT, OUT, dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--start-ticks',required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    state_path=OUT/f'livo_full_{a.tag}_wait_status.json';assert not state_path.exists()
    def alive():
        try:
            proc=Path('/proc')/str(a.pid);raw=(proc/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
            if fields[19]!=a.start_ticks or fields[0]=='Z':return False
            cmd=(proc/'cmdline').read_bytes().split(b'\0')
            assert str(a.input).encode() in cmd and any(x.endswith(b'/prepare_vbot_bag.py') for x in cmd)
            return True
        except (FileNotFoundError,ProcessLookupError):return False
    state=dict(complete=False,pid=os.getpid(),converter_pid=a.pid,converter_start_ticks=a.start_ticks,input=str(a.input),tag=a.tag)
    def publish(phase):state.update(phase=phase,updated_epoch=time.time());dump(state_path,state)
    try:
        while alive():publish('waiting_for_verified_converter');time.sleep(2)
        assert (a.input/'input_audit.json').exists(),'converter exited without complete audit; do not restart automatically'
        publish('launching_full_validation')
        command=[sys.executable,str(ROOT/'scripts/benchmarks/run_livo_full_validation.py'),'--input',str(a.input),'--tag',a.tag]
        child=subprocess.Popen(command,cwd=ROOT);state['validation_pid']=child.pid;publish('running_full_validation')
        rc=child.wait();state['returncode']=rc
        if rc:raise RuntimeError(f'full validation exited {rc}; inspect preserved logs')
        state['complete']=True;publish('execution_complete_pending_result_audit')
    except Exception as e:state['error']=repr(e);publish('failed');raise


if __name__=='__main__':main()
