"""Verify paper freeze hashes without loading models or running experiments."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/paper/frozen_v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    m=json.loads((DEST/'manifest.json').read_text());fail=[]
    for entry in m['files']:
        p=ROOT/entry['path']
        if not p.is_file() or digest(p)!=entry['sha256']:fail.append(entry['path'])
    archive=DEST/'source_snapshot.tar.gz'
    if digest(archive)!=m['archive_sha256']:fail.append('source archive checksum')
    with tarfile.open(archive,'r:gz') as tf:
        archived={x.name:x for x in tf.getmembers()}
        expected={r['path'] for r in m['files'] if r['in_source_snapshot']}
        if set(archived)!=expected:fail.append('archive member set')
        for r in m['files']:
            if r['in_source_snapshot']:
                f=tf.extractfile(r['path'])
                if hashlib.sha256(f.read()).hexdigest()!=r['sha256']:fail.append('archive:'+r['path'])
    protocol=json.loads((DEST/'protocol.json').read_text())
    assert len(protocol['variants'])==8 and len(protocol['final_test_scenes'])==6
    assert len({v['geometry'] for v in protocol['variants']})==5
    assert next(v for v in protocol['variants'] if v['id']=='M')['category']=='consensus_sqrt_area'
    assert protocol['max_mapping_runs']==30 and not protocol['experiments_started_by_this_freeze']
    if fail:raise SystemExit('FREEZE MISMATCH\n'+'\n'.join(fail))
    print(f"PASS: {len(m['files'])} local files; {len(expected)} archived files; 8 variants / 6 scenes / 30 map limit.")
    print('External data / weights are references, not included or re-hashed by this lightweight check.')


if __name__=='__main__':main()
