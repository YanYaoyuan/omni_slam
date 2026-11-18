#!/usr/bin/env python3
"""Download a pinned public NTU archive with validated ranges and source MD5."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--revision',required=True)
    ap.add_argument('--sequence',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    entry=next(x for x in json.loads(args.manifest.read_text())['files'] if x['filename']==args.sequence+'.zip')
    final=args.output/entry['filename']
    if final.exists():raise FileExistsError(final)
    url=f"https://huggingface.co/datasets/Yuan-Shenghai/NTU-VIRAL-DATASET/resolve/{args.revision}/{entry['path']}?download=true"
    directory=args.output/(args.sequence+'_parts');directory.mkdir(exist_ok=True)
    part_size=256<<20;total=entry['size'];began=time.monotonic()
    def part(i):
        lo=i*part_size;hi=min(total,(i+1)*part_size)-1;path=directory/f'{i:03d}.part'
        for attempt in range(4):
            have=path.stat().st_size if path.exists() else 0
            if have==hi-lo+1:return path
            if have>hi-lo+1:raise ValueError('oversized cached range')
            try:
                with requests.get(url+f'&segment={i}',headers={'Range':f'bytes={lo+have}-{hi}'},stream=True,timeout=(30,90)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {lo+have}-{hi}/{total}':
                        raise ValueError('server did not honor exact range')
                    with path.open('ab') as f:
                        for block in r.iter_content(1<<20):f.write(block)
                if path.stat().st_size!=hi-lo+1:raise ValueError('incomplete range')
                print('range complete',i,flush=True);return path
            except Exception:
                if attempt==3:raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts=list(pool.map(part,range((total+part_size-1)//part_size)))
    md5=hashlib.md5();sha=hashlib.sha256();temporary=final.with_suffix('.zip.assembling')
    with temporary.open('wb') as target:
        for path in parts:
            with path.open('rb') as source:
                while data:=source.read(8<<20):
                    target.write(data);md5.update(data);sha.update(data)
    if md5.hexdigest()!=entry['source_md5']:raise ValueError('official source MD5 mismatch')
    temporary.rename(final)
    for path in parts:path.unlink()
    directory.rmdir()
    report=dict(source=entry,revision=args.revision,md5=md5.hexdigest(),sha256=sha.hexdigest(),wall_s=time.monotonic()-began)
    (args.output/(args.sequence+'_download.json')).write_text(json.dumps(report,indent=2))
    gt_url=f'https://huggingface.co/datasets/Yuan-Shenghai/NTU-VIRAL-DATASET/resolve/{args.revision}/ground_truth/{args.sequence}/ground_truth.csv'
    r=requests.get(gt_url,timeout=60);r.raise_for_status()
    (args.output/(args.sequence+'_ground_truth.csv')).write_bytes(r.content)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
