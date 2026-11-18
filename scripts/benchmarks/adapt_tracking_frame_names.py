"""Expose unchanged PNG bytes to the native tracker's JPEG-name-only loader."""
import shutil
from pathlib import Path
from PIL import Image
from paper_final_common import read,dump,sha
from probe_world_anchor_tracking import OUT,verify,prepared


def main():
    verify();d=prepared();assert not (OUT/'predictions_complete.json').exists()
    assert not list((OUT/'cases').rglob('*.npz')),'This adaptation must precede all inference outputs'
    log=OUT/'infer.log';assert 'no images found' in log.read_text();shutil.copyfile(log,OUT/'infer_initial.log');aliases=[]
    for case in d['cases']:
        for j,record in enumerate(case['records']):
            original=Path(record['rgb']);alias=Path(case['video'])/f'{j:06d}.jpg';assert not alias.exists();alias.symlink_to(original)
            assert sha(alias)==sha(original) and Image.open(alias).format=='PNG'
            aliases.append(dict(alias=str(alias),original=str(original),sha256=sha(original)))
    dump(OUT/'runtime_input_adapter.json',dict(complete=True,source_sha256=sha(__file__),aliases=aliases,
        reason='Native Sam3TrackerPredictor imports utils/sam2_utils JPEG-extension enumerator. First run stopped during first frame loading before any image encode/mask. Add .jpg symbolic names resolving to ORIGINAL PNG bytes; PIL sniffs format; no conversion/recompression or changed pixels. Existing .png aliases ignored by this enumerator.',
        initial_log_sha256=sha(OUT/'infer_initial.log'),protocol_sha256=sha(OUT/'protocol.json'),prepared_sha256=sha(OUT/'prepared.json'),algorithm_change=False,first_attempt_neural_image_encodes=0))
    print('UNCHANGED INPUT ALIASES',len(aliases),flush=True)


if __name__=='__main__':main()
