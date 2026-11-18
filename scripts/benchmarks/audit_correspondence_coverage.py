"""Posthoc attrition diagnosis of the sealed nine-clip correspondence audit.

No prediction or primary protocol changes. Missing depth support is NOT a
negative visibility label. Ungated semantic readout is explicitly secondary.
"""
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from probe_observation_correspondence import OUT, inside
from readout_observation_correspondence import semantic_stats, METHODS, DEST


def main():
    assert not (OUT/'coverage_audit.json').exists()
    prep=read(OUT/'prepared.json');pred=read(OUT/'predictions_complete.json');primary=read(OUT/'readout.json')
    inputs={str(OUT/n):sha(OUT/n) for n in ['protocol.json','prepared.json','predictions_complete.json','readout.json']}
    for f,h in {**prep['files'],**pred['files'],**primary['files']}.items():assert sha(f)==h,f
    rows=[];pooled={};totals={};replayed=0
    for r in prep['results']:
        d=np.load(r['input']);o=np.load(OUT/'predictions'/(r['name']+'.npz'));a=np.load(OUT/'readout'/(r['name']+'.npz'))
        n=len(d['source_ids']);xy=d['seed_xy'];native=d['seed_native'];size=d['native_size']
        np.testing.assert_allclose((native+.5)/d['scale']-.5,xy,rtol=0,atol=1e-12)
        # Independently reconstruct full-clip acceptance with scalar loops.
        for i in range(n):
            good=True
            for frame in range(3):
                for direction in ['', '_reverse']:
                    coords=o['vggt_native' if not direction else 'vggt_reverse'][frame,i]
                    good &= bool(np.isfinite(coords).all() and 0<=coords[0]<size[0] and 0<=coords[1]<size[1])
                    good &= bool(o['vggt'+direction+'_vis'][frame,i]>.5 and o['vggt'+direction+'_conf'][frame,i]>.5)
            cycle=float(np.sqrt(np.sum((o['vggt_reverse'][-1,i]-native[i])**2)))
            good &= bool(cycle<=2)
            assert good==bool(o['vggt_valid'][i])
            replayed+=1
        geometry_inside=a['geometry_class']>=0
        nonseed=np.ones((3,n),bool);nonseed[0]=False
        own=nonseed & geometry_inside & o['vggt_valid'][None]
        common=own & o['lk_valid'][None]
        # Secondary domain lacks independent current-depth confirmation.
        missing=own & ~d['measured_visible']
        labs={m:a[m+'_class'] for m in METHODS};t=np.tile(a['truth'],(3,1));c=np.tile(a['confidence'],(3,1))
        row=dict(name=r['name'],key=r['key'],seeds=n,assigned=int((d['owner']>0).sum()),
            nonseed_projected_in_image=int((nonseed&geometry_inside).sum()),
            same_source_depth_confirmed=int(a['base'].sum()),
            depth_confirmed_with_reference=int((a['base']&(t>=0)).sum()),
            native_valid_seeds=int(o['vggt_valid'].sum()),lk_valid_seeds=int(o['lk_valid'].sum()),
            native_accepted_nonseed=int(own.sum()),native_accepted_without_current_depth=int(missing.sum()),
            primary_common=int(a['common'].sum()),primary_common_with_reference=int((a['common']&(t>=0)).sum()),
            native_forward_visibility_all=int((o['vggt_vis']>.5).all(0).sum()),
            native_forward_confidence_all=int((o['vggt_conf']>.5).all(0).sum()),
            native_cycle_accepted=int((o['vggt_cycle']<=2).sum()))
        rows.append(row)
        pooled.setdefault(r['key'],[]).append(dict(labels=labs,truth=t,confidence=c,common=common,own=own,missing=missing))
    for key,group in pooled.items():
        labs={m:np.concatenate([r['labels'][m].ravel() for r in group]) for m in METHODS}
        t=np.concatenate([r['truth'].ravel() for r in group]);c=np.concatenate([r['confidence'].ravel() for r in group])
        sums={k:sum(r[k] for r in rows if r['key']==key) for k in rows[0] if k not in ['name','key']}
        diagnostics={}
        for name in ['common','own','missing']:
            domain=np.concatenate([r[name].ravel() for r in group]);s=semantic_stats(labs,t,c,domain)
            # Independent Python accumulation verifies the vector readout.
            eligible=[i for i in range(len(t)) if domain[i] and 0<=t[i]<19 and c[i]>0]
            assert len(eligible)==s['observations']
            for method in METHODS:
                mass=sum(float(c[i]) for i in eligible)
                score=sum(float(c[i]) for i in eligible if labs[method][i]==t[i])/mass if mass else None
                expected=s['methods'][method]['accuracy']
                assert (score is None and expected is None) or abs(score-expected)<1e-12
            diagnostics[name]=s
        totals[key]=dict(counts=sums,secondary_semantics_without_depth_confirmation=diagnostics,
            primary_disposition='not_identifiable_no_common_labeled_observations' if key.startswith('kitti') else 'robot_no_ground_truth')
    for f,h in inputs.items():assert sha(f)==h
    result=dict(complete=True,created_epoch=time.time(),source_sha256=sha(__file__),inputs=inputs,rows=rows,totals=totals,
        validity_replays=replayed,new_model_calls=0,new_maps=0,new_ap=0,
        primary_protocol_unchanged=True,primary_screen_gate=primary['screen_gate'],
        interpretation='The original sparse-depth-confirmed comparison is not identifiable, not evidence of tracker inferiority. Secondary no-current-depth readout is posthoc on exactly the same sealed predictions, not a replacement primary gate, visibility truth, instance accuracy or permission to use missing measurements as negatives. Native point transport did not correct any geometry-readout class error on its valid known-source domain in either public window; this does not rule out same-class instance association benefits, which were not tested.',
        decision='Do not replace measured geometry projections with tracked coordinates or generate full-map variants from this probe. Freeze this configuration, no selection/threshold/model sweep. Future mechanism must define and preflight observable object-level evidence before inference, including missing-depth handling; novelty cannot be native tracking hookup.')
    dump(OUT/'coverage_audit.json',result)
    import html,json
    page=DEST/'index.html';old=page.read_text()
    page.write_text(old+'<h2>Coverage audit: primary comparison is not identifiable</h2><p>There are zero common depth-confirmed labeled observations in each public window. Null primary accuracy is NOT a zero score or proof of tracker failure. The following no-current-depth diagnostics are posthoc, not the primary experiment; source semantic agreement does not establish physical visibility or instance identity.</p><pre>'+html.escape(json.dumps(totals,indent=2))+'</pre>')
    dump(OUT/'delivery_audit.json',dict(complete=True,validity_replays=replayed,half_pixel_roundtrip=True,
        primary_inputs_unchanged=True,secondary_scalar_scores_replayed=True,
        report_sha256=sha(page),coverage_sha256=sha(OUT/'coverage_audit.json'),
        figures=primary['figures'],new_model_calls=0,new_maps=0,new_ap=0,goal_complete=False))
    print('CORRESPONDENCE AUDIT',totals,flush=True)


if __name__=='__main__':main()
