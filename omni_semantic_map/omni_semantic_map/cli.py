import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .objects import query, reproject_map


def main():
    parser = argparse.ArgumentParser(
        description="Query or update keyframe-anchored semantic objects"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    q = sub.add_parser("query")
    q.add_argument("map", type=Path)
    q.add_argument("text")
    q.add_argument("--include-tentative", action="store_true")
    r = sub.add_parser("reproject")
    r.add_argument("map", type=Path)
    r.add_argument("--poses", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.map.read_text())
    if args.command == "query":
        result = {
            "frame": document["frame"],
            "map_revision": document["map_revision"],
            "query": args.text,
            "objects": query(document, args.text, args.include_tentative),
        }
        for obj in result["objects"]:
            obj["evidence"] = str((args.map.parent / obj["evidence"]).resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if args.output.exists():
            parser.error("output exists; retain previous map revision")
        archive = Path(document["keyframe_source"]) / "keyframes.csv"
        if (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            != document["keyframe_index_sha256"]
        ):
            parser.error("source keyframe archive has changed")
        rows = np.atleast_1d(np.genfromtxt(archive, delimiter=",", names=True))
        revision = np.load(args.poses)
        if not np.array_equal(revision["stamps"], rows["stamp"]):
            parser.error("pose revision does not match this map's keyframe epochs")
        poses = revision["after"]
        if poses.shape != (len(rows), 4, 4) or not np.isfinite(poses).all():
            parser.error("invalid keyframe poses")
        result = reproject_map(
            document, poses, hashlib.sha256(args.poses.read_bytes()).hexdigest()
        )
        # Evidence remains resolvable if the new revision lives elsewhere.
        for obj in result["objects"]:
            for obs in obj["observations"]:
                obs["evidence"] = str((args.map.parent / obs["evidence"]).resolve())
            obj["evidence"] = str((args.map.parent / obj["evidence"]).resolve())
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
