"""Posthoc comparison to existing source ownership, not new map variants."""
import numpy as np
from paper_final_common import read, dump, sha
from probe_object_surface_refresh import OUT


def main():
    assert not (OUT/'existing_support_audit.json').exists()
    prep=read(OUT/'prepared.json');result=read(OUT/'readout.json');rows={}
    for key,cases in result['results'].items():
        selected={r['object_id']:r for r in prep['results'][key]['selected']};rows[key]=[]
        for case in cases:
            oid=case['object_id'];domain=np.load(selected[oid]['input']);current=domain['owner']==oid
            r=dict(object_id=oid,diagnostic_eligible=case['diagnostic_eligible'],current_source_iou=None,branches={})
            for branch in ['full','crop']:
                data=np.load(OUT/'readout'/key/f'object_{oid}_{branch}.npz')
                np.testing.assert_array_equal(data['source_ids'],domain['source_ids'])
                mask=data['selected'];known=data['known'];weight=data['confidence']
                row=dict(new_unassigned=int((mask&(domain['owner']==0)).sum()),
                         overlaps_other_owner=int((mask&(domain['owner']>0)&~current).sum()),
                         omits_current=int((~mask&current).sum()))
                if case['diagnostic_eligible']:
                    target=data['truth']==case['target_gt']
                    def iou(selection):
                        return float(weight[selection&target].sum()/max(1e-12,weight[known&(selection|target)].sum()))
                    r['current_source_iou']=iou(current)
                    row.update(selected_mask_iou=iou(mask),new_correct=int((mask&(domain['owner']==0)&target).sum()),
                        new_other_known=int((mask&(domain['owner']==0)&known&~target).sum()),
                        new_unknown=int((mask&(domain['owner']==0)&~known).sum()),
                        omitted_current_correct=int((~mask&current&target).sum()),
                        overlaps_other_owner_correct=int((mask&(domain['owner']>0)&~current&target).sum()))
                r['branches'][branch]=row
            rows[key].append(r)
    dump(OUT/'existing_support_audit.json',dict(complete=True,results=rows,source_sha256=sha(__file__),
        prepared_sha256=sha(OUT/'prepared.json'),readout_sha256=sha(OUT/'readout.json'),
        scope='Posthoc readout of SAME sealed masks and common source domain versus existing source ownership. No composed map, new selection, threshold, new candidate or new AP. Source ownership proxy is not the fused-coordinate final map.',
        aggregate={key:{branch:dict(
            eligible=sum(r['diagnostic_eligible'] for r in rs),
            higher_than_existing=sum(r['diagnostic_eligible'] and r['branches'][branch]['selected_mask_iou']>r['current_source_iou']+1e-12 for r in rs),
            new_correct=sum(r['branches'][branch].get('new_correct',0) for r in rs),
            new_other_known=sum(r['branches'][branch].get('new_other_known',0) for r in rs),
            omitted_current_correct=sum(r['branches'][branch].get('omitted_current_correct',0) for r in rs))
            for branch in ['full','crop']} for key,rs in rows.items()}))
    print(read(OUT/'existing_support_audit.json')['aggregate'])


if __name__=='__main__':main()
