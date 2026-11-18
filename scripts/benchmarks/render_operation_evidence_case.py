"""Improve layout of a real-mask audit figure without changing audited data."""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_final_common import ROOT,read,dump,sha

REPORT=ROOT/'output/operation_evidence_audit_v1'
audit=read(REPORT/'summary.json');case=audit['case'];images=case['image_provenance']
fig,axes=plt.subplots(2,4,figsize=(16,9),layout='constrained')
for i,(ax,item) in enumerate(zip(axes.flat,images)):
    assert sha(item['rgb'])==item['rgb_sha256'] and sha(item['mask_file'])==item['mask_sha256']
    cache=Path(item['mask_file']).parents[1];manifest=read(cache/'manifest.json')
    rec=next(r for r in manifest['records'] if r['frame_id']==item['frame_id']);local=item['observation_id']-rec['observation_offset']
    rgb=np.asarray(Image.open(item['rgb']).convert('RGB'))
    with np.load(item['mask_file']) as f:mask=np.unpackbits(f['masks_packed'][local],axis=1)[:,:rgb.shape[1]].astype(bool)
    layer=np.zeros((*mask.shape,4));layer[mask]=[.1,.85,.75,.45] if i<4 else [1.,.2,.45,.45]
    ax.imshow(rgb);ax.imshow(layer);oid=list(case['groups'])[i//4]
    ax.set_title(f"Object {oid} | frame {item['frame_id']}\nobservation {item['observation_id']}",fontsize=10,pad=12);ax.axis('off')
fig.suptitle('All retained masks of the two incorrectly merged objects',fontsize=15)
path=REPORT/'harmful_merge_all_masks.png';fig.savefig(path,dpi=130);plt.close(fig)
dump(REPORT/'figure_verification.json',dict(source_sha256=sha(__file__),audit_sha256=sha(REPORT/'summary.json'),
    output_sha256=sha(path),verified_images=len(images),layout_only=True))
