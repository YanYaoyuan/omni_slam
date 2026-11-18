"""One final sampling correction; original failed screening remains immutable."""
import argparse,inspect,time
from pathlib import Path
from paper_final_common import ROOT,read,dump,sha
from run_intrinsic_public_holdout import isolated_adapter

PARENT=ROOT/'build/benchmarks/surface_continuation_probe_v1'
OUT=ROOT/'build/benchmarks/foreground_balanced_continuation_probe_v1'
OLD="selected=np.r_[rng.choice(np.flatnonzero(target==0),count,replace=False),rng.choice(np.flatnonzero(target==1),count,replace=False)]"
NEW="positive_bg=np.flatnonzero(train['pair_kind']==0);positive_fg=np.flatnonzero(train['pair_kind']==1);assert len(positive_bg)>=count//2 and len(positive_fg)>=count-count//2;selected=np.r_[rng.choice(np.flatnonzero(target==0),count,replace=False),rng.choice(positive_bg,count//2,replace=False),rng.choice(positive_fg,count-count//2,replace=False)]"


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(f)==h,f
    return p


def freeze():
    assert not OUT.exists();n=isolated_adapter('probe_surface_continuation.py');n.verify();s=read(PARENT/'summary.json');assert s['complete'] and not s['screening_gate'];assert inspect.getsource(n.train).count(OLD)==1
    p=read(PARENT/'protocol.json');p.update(created_epoch=time.time(),parent_failed_gate=False,parent_summary_sha256=sha(PARENT/'summary.json'),
        correction='One final predeclared correction of same-owner positive sampling: half same-foreground, half same-background; same total positives and negative sample,architecture,epochs,optimizer,seed,all-known-edge test and original gate. No other balance/weight/threshold variants.',
        rationale='Generic positive class was about90percentbackground; first fit harmed same-foreground recall on0002 despite better overallAUROC. Foreground preservation is a fixed task requirement, not a newly relaxed metric.',
        budget='4additionalfits;8totalincludingfailedparent.0newfeatures/maps/modelinference/reference reads. If this originalgatefails,close surface-affinity candidate without further fit/search or maps.',
        interpretation='Development correction after pseudo-label screening, not independent method validation. Original failure remains false and parentfilesunchanged.')
    p['source_hashes'][str(Path(__file__))]=sha(__file__)
    for path in [PARENT/'protocol.json',PARENT/'summary.json',PARENT/'features_complete.json']:p['input_hashes'][str(path)]=sha(path)
    dump(OUT/'protocol.json',p);(OUT/'features').symlink_to(PARENT/'features',target_is_directory=True);dump(OUT/'features_complete.json',read(PARENT/'features_complete.json'))
    print('ONE FOREGROUND-BALANCED CORRECTION FROZEN',flush=True)


def train():
    verify();n=isolated_adapter('probe_surface_continuation.py');n.OUT=OUT;n.verify=verify;source=inspect.getsource(n.train);assert source.count(OLD)==1;source=source.replace(OLD,NEW);namespace=n.__dict__;exec(compile(source,str(__file__)+':sampling_correction','exec'),namespace);namespace['train']()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','train']);globals()[p.parse_args().action]()
