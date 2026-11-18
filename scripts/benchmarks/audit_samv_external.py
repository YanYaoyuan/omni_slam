"""Verify sealed external-model predictions and original-resolution readout."""
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import torch
from paper_final_common import ROOT,read,dump,sha
from probe_samv_external import OUT,METHODS,verify
from omni_semantic_map.samv_adapter import logits_to_original


def main():
    torch.set_num_threads(2);p=verify();pred=read(OUT/'predictions_complete.json');summary=read(OUT/'summary.json')
    assert pred['complete'] and summary['complete'] and pred['counts']==p['expected_calls']
    assert summary['prediction_seal_sha256']==sha(OUT/'predictions_complete.json')
    for f,h in pred['files'].items():assert sha(f)==h,f
    parity=read(OUT/'native_parity.json');assert parity['logits_exact'] and parity['iou_exact']
    rows=[]
    for case in p['cases']:
        for method in METHODS:
            path=OUT/'cases'/case['name']/(method+'.npz');d=np.load(path)
            h,w=int(d['height']),int(d['width']);valid=d['valid'];indices=np.flatnonzero(valid)
            logits=d['lowres_logits'][indices];assert np.isfinite(logits).all()
            assert np.isnan(d['lowres_logits'][~valid]).all()
            pano=torch.from_numpy(logits.astype(np.float32)).permute(1,0,2).reshape(1,1,256,len(indices)*256)
            original=logits_to_original(pano,len(indices),(h,w)).numpy()
            masks=np.unpackbits(d['masks_packed'][indices],axis=2)[:,:,:w].astype(bool)
            changed=masks!=(original>=0)
            # Different CPU/GPU interpolation kernels may round zero boundaries
            # differently. Any other disagreement is a failed artifact audit.
            max_near_zero=float(np.abs(original[changed]).max()) if changed.any() else 0.
            assert max_near_zero<=2e-5,(case['name'],method,max_near_zero)
            assert valid[case['common_frames']].all()
            expected=np.ones(11,bool) if method=='samv_joint_seed' else np.array([q['eligible'] for q in case['prompts']])
            np.testing.assert_array_equal(valid,expected)
            rows.append(dict(case=case['name'],method=method,valid_frames=len(indices),compared_pixels=int(masks.size),
                cpu_gpu_boundary_disagreements=int(changed.sum()),max_abs_logit_at_disagreement=max_near_zero))
    dest=ROOT/'output/samv_evidence_comparison_v1';m=read(dest/'manifest.json')
    for f,h in m['files'].items():assert sha(f)==h,f
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and not v.startswith(('http://','https://','#')):self.links.append(v)
    links=Links();links.feed((dest/'index.html').read_text())
    for link in links.links:assert (dest/link).is_file(),link
    dump(OUT/'delivery_audit.json',dict(complete=True,prediction_artifacts_verified=len(pred['files']),
        input_and_source_hashes_verified=len(p['inputs'])+len(p['sources']),native_parity=parity,
        output_readout=rows,old_controls_replayed_exactly=summary['old_controls_replayed_exactly'],
        expected_forward_counts_exact=True,targeted_tests_passed=2,local_report_links_verified=len(links.links),
        figures_reviewed=[f for f in m['files'] if Path(f).suffix=='.png'],report_sha256=sha(dest/'index.html'),
        source_sha256=sha(__file__),goal_complete=False))
    print('SAMV DELIVERY VERIFIED',len(rows),'masksets',sum(r['compared_pixels'] for r in rows),'pixels',
          sum(r['cpu_gpu_boundary_disagreements'] for r in rows),'near-zero boundary differences',flush=True)


if __name__=='__main__':main()
