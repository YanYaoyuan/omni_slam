"""CRC-verified bounded downloads for preselected KITTI-360 windows.

Reserved annotation PLYs remain opaque: byte copying does not decode labels.
Separate remote ZIP streams per worker; original large archives are not saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
import time
import zipfile
from paper_final_common import ROOT, read, dump, sha
from prepare_hypersim_subset import RangeFile

BASE = ROOT/'build/datasets/kitti360_prepare'
OUT = ROOT/'build/datasets/kitti360_windows_v1'
URL = 'https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/'


def extract(z, member, target):
    info = z.getinfo(member); path = target/member
    if not path.resolve().is_relative_to(target.resolve()) or info.file_size > 256*1024**2:
        raise ValueError('Unsafe or oversized member')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        crc = 0; size = 0
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024**2), b''): crc = zipfile.crc32(b, crc); size += len(b)
        if size != info.file_size or crc & 0xffffffff != info.CRC: raise ValueError('Existing file checksum mismatch')
    else:
        temporary = path.with_suffix(path.suffix+'.partial')
        with z.open(info) as source, temporary.open('wb') as dest:
            while True:
                b = source.read(1024**2)
                if not b: break
                dest.write(b)
        assert temporary.stat().st_size == info.file_size
        temporary.replace(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha(path), crc32=info.CRC)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=3); args = parser.parse_args()
    if not 1 <= args.workers <= 3: raise ValueError('At most three independent remote archives')
    selection = read(BASE/'window_selection.json')
    for path, want in selection['source_sha256'].items(): assert sha(ROOT/path) == want
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = read(BASE/'semantic_archive_index.json'); jobs = []
    for row in selection['windows']:
        for kind, suffix, extension in [('data_2d_raw','image_00','.png'),('data_3d_raw','velodyne','.bin')]:
            jobs.append(dict(name=row['scene']+'_'+suffix, url=URL+kind+'/'+row['scene']+'_'+suffix+'.zip',
                kind=kind, row=row, extension=extension))
    jobs.append(dict(name='opaque_references',url=metadata['url'],kind='reference'))
    state = dict(pid=os.getpid(), complete=False, failed=False, completed=[], started_epoch=time.time(),
                 selection_sha256=sha(BASE/'window_selection.json'),script_sha256=sha(__file__))
    dump(OUT/'download_status.json', state)
    def worker(job):
        final = OUT/(job['name']+'_download.json')
        if final.exists():
            old = read(final); assert old['selection_sha256'] == state['selection_sha256']
            for f in old['files']: assert sha(ROOT/f['path']) == f['sha256']
            return job['name']
        stream = RangeFile(job['url'], 256*1024**2)
        with zipfile.ZipFile(stream) as z:
            if job['kind'] == 'reference': members = [r['member'] for r in selection['windows']]; target = OUT/'opaque_references'
            else:
                basenames = {f"{f:010d}{job['extension']}" for f in job['row']['input_frames']}
                members = [i.filename for i in z.infolist() if Path(i.filename).name in basenames]
                assert len(members) == len(basenames)
                target = OUT/job['kind']
            files = []
            for member in members:
                files.append(extract(z,member,target))
                dump(OUT/(job['name']+'_progress.json'),dict(files_done=len(files),files_expected=len(members),range_bytes=stream.used,updated_epoch=time.time()))
        dump(final,dict(complete=True,url=job['url'],selection_sha256=state['selection_sha256'],files=files,
                        downloaded_range_bytes=stream.used,archive_bytes=stream.size,reference_labels_decoded=False))
        return job['name']
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(worker,job):job['name'] for job in jobs}
        for future in as_completed(tasks):
            try: name = future.result(); state['completed'].append(name); print('DONE',name,flush=True)
            except Exception as error:
                errors.append(dict(job=tasks[future],error=repr(error))); print('FAILED',errors[-1],flush=True)
            state.update(errors=errors,failed=bool(errors),updated_epoch=time.time());dump(OUT/'download_status.json',state)
    state.update(complete=not errors,finished_epoch=time.time());dump(OUT/'download_status.json',state)
    if errors: raise RuntimeError('Inspect failed downloads before resuming')


if __name__ == '__main__': main()
