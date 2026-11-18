"""Run the frozen CPU backend/evaluation once each frontend cache is complete."""
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from audit_evidence_retention import ROOT, OUT, REPORT, dump
from prepare_retention_holdout import SCENES,DATA
from omni_semantic_map.flat_benchmark import sha256


def main():
    protocol=json.loads((OUT/'stage3_evaluation_protocol.json').read_text())
    for file,h in protocol['source_sha256'].items():assert sha256(ROOT/file)==h
    equivalence=json.loads((OUT/'backend_equivalence_v2/office_0/run.json').read_text())
    assert equivalence['complete'] and equivalence['original_backend_geometry_memberships_and_mean_classes_exact']
    state=dict(complete=False,pid=os.getpid(),completed=[]);status=OUT/'holdout_evaluation_status.json'
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/omni-retention-mpl',PYTHONUNBUFFERED='1')
    for scene in SCENES:
        cache=DATA/'strong_frontend/v1'/scene/'cache';run=OUT/'heldout_runs'/scene;evaluation=OUT/'heldout_evaluation'/scene
        state.update(scene=scene,phase='waiting_for_cache');dump(status,state)
        while not (cache/'manifest.json').exists() or not json.loads((cache/'manifest.json').read_text())['complete']:
            queue=json.loads((OUT/'split_queue_status.json').read_text())
            if queue['state']=='failed':raise RuntimeError('frontend queue failed; inspect before any retry')
            time.sleep(5)
        commands=[('backend',ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python',
            ROOT/'scripts/benchmarks/retention_holdout_backend.py',['--cache',str(cache),'--output',str(run)]),
            ('evaluation',ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python',
            ROOT/'scripts/benchmarks/evaluate_retention_holdout.py',['--scene',scene,'--cache',str(cache),'--run',str(run),'--output',str(evaluation)])]
        for phase,python,script,arguments in commands:
            state.update(phase=phase);dump(status,state)
            with (OUT/'logs'/(scene+'_'+phase+'.log')).open('w') as log:
                child=subprocess.Popen([str(python),str(script),*arguments],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;dump(status,state);rc=child.wait()
            if rc:
                state.update(phase='failed',returncode=rc);dump(status,state);raise SystemExit(rc)
        metrics=json.loads((evaluation/'metrics.json').read_text());assert metrics['complete']
        state['completed'].append(scene);dump(status,state)
        print('SCORED',scene,{k:v['observed_voxel_instance_pq']*100 for k,v in metrics['methods'].items()},flush=True)
    results=[json.loads((OUT/'heldout_evaluation'/s/'metrics.json').read_text()) for s in SCENES]
    summary=dict(complete=True,scenes=SCENES,methods={},input_domains={r['scene']:r['domain'] for r in results},
        evaluation_protocol_sha256=sha256(OUT/'stage3_evaluation_protocol.json'),scope='three previously unused SceneNN scenes, frozen paired diagnostic; not official leaderboard')
    for name in ['mean','area_density']:
        scores=[r['methods'][name] for r in results]
        summary['methods'][name]=dict(pq_by_scene=[r['observed_voxel_instance_pq'] for r in scores],
            macro_pq=float(np.mean([r['observed_voxel_instance_pq'] for r in scores])),totals={key:sum(r[key] for r in scores) for key in ['tp','fp','fn']})
    delta=np.array(summary['methods']['area_density']['pq_by_scene'])-summary['methods']['mean']['pq_by_scene']
    summary['paired']=dict(mean_delta_pq100=float(delta.mean()*100),delta_pq100=(delta*100).tolist(),
        wins=int((delta>1e-8).sum()),ties=int((abs(delta)<=1e-8).sum()),losses=int((delta < -1e-8).sum()))
    dump(REPORT/'stage3/heldout_summary.json',summary)
    state.update(complete=True,phase='complete',child_pid=None);dump(status,state)
    print(json.dumps(summary['paired']),flush=True)


if __name__=='__main__':main()
