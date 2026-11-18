"""Prepare metadata roots before the frozen holdout's candidates stage.

Stage reuse requires a protocol copy in each output root. This adds only
metadata; it cannot change feature extraction, trained weights or predictions.
Run after maps and before candidates. Existing protocol copies must agree.
"""
import shutil
from paper_final_common import dump,sha
from run_intrinsic_public_holdout import OUT,verify,sealed


def main():
    verify(True);sealed('base_maps_complete');files={}
    for name in ['ownership_axis','ownership_intrinsic']:
        root=OUT/name;root.mkdir(exist_ok=True);target=root/'protocol.json'
        if target.exists():assert sha(target)==sha(OUT/'protocol.json')
        else:shutil.copyfile(OUT/'protocol.json',target)
        files[str(target)]=sha(target)
    dump(OUT/'adapter_setup.json',dict(complete=True,files=files,setup_source_sha256=sha(__file__),scope='Metadata only. No algorithm/input/model change.'))
    print('Adapter metadata ready',flush=True)


if __name__=='__main__':main()
