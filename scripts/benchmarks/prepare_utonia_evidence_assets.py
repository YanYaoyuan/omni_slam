"""Acquire one pinned official Utonia encoder; no labels, model search or fit."""
import hashlib
import os
from pathlib import Path
import subprocess
import time
import requests
from huggingface_hub import HfApi
from paper_final_common import ROOT, dump, sha

OUT = ROOT / 'build/benchmarks/utonia_evidence_v1'


def main():
    assert not (OUT / 'assets.json').exists()
    repo = OUT / 'upstream'
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    info = HfApi().model_info('Pointcept/Utonia', files_metadata=True)
    entry = next(s for s in info.siblings if s.rfilename == 'utonia.pth')
    expected = entry.lfs.sha256; size = entry.size
    url = f'https://huggingface.co/Pointcept/Utonia/resolve/{info.sha}/utonia.pth'
    folder = OUT / 'weights'; folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'utonia.pth'; partial = folder / 'utonia.pth.part'
    dump(OUT / 'acquisition.json', dict(started_epoch=time.time(), repo='Pointcept/Utonia', revision=info.sha,
        weight_url=url, expected_size=size, expected_sha256=expected, source_commit=commit))
    print('UTONIA RELEASE', info.sha, size, expected, flush=True)
    if not target.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        # Strict, independently ranged chunks make transient transfers resumable.
        chunk = 16 * 1024 * 1024
        with partial.open('ab') as output:
            while offset < size:
                end = min(offset + chunk, size) - 1
                for attempt in range(4):
                    try:
                        response = requests.get(url, headers={'Range': f'bytes={offset}-{end}'}, timeout=(20, 90))
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError(f'Expected range206, received{response.status_code}')
                        if response.headers.get('Content-Range') != f'bytes {offset}-{end}/{size}':
                            raise RuntimeError('Unexpected Content-Range')
                        data = response.content
                        if len(data) != end - offset + 1: raise RuntimeError('Incomplete range')
                        output.write(data); output.flush(); os.fsync(output.fileno())
                        offset = end + 1
                        print('UTONIA BYTES', offset, '/', size, flush=True)
                        break
                    except Exception as error:
                        print('UTONIA RANGE RETRY', offset, attempt + 1, type(error).__name__, flush=True)
                        if attempt == 3: raise
                        time.sleep(2)
        assert partial.stat().st_size == size and sha(partial) == expected
        partial.replace(target)
    assert target.stat().st_size == size and sha(target) == expected
    files = {str(f): sha(f) for f in repo.rglob('*.py') if '.git' not in f.parts}
    files[str(target)] = expected
    dump(OUT / 'assets.json', dict(complete=True, files=files, source_commit=commit, repo='Pointcept/Utonia',
        checkpoint_revision=info.sha, checkpoint=str(target), checkpoint_bytes=size, checkpoint_sha256=expected,
        source_script_sha256=sha(__file__), local_training=0, weights='Official base encoder only; no ScanNet head, small/tiny variants or checkpoint search.'))
    print('UTONIA ASSETS COMPLETE', flush=True)


if __name__ == '__main__': main()
