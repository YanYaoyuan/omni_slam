#!/usr/bin/env python3
"""Read-only rectified image evidence for the physical epochs of audited revisits."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_vbot_bag import time_ns


def nearest_image(entry, target, maps, seek=True):
    codec = av.CodecContext.create("hevc", "r")
    codec.thread_count = 1
    decoder_factory = DecoderFactory()
    decoders = {}
    target_ns = round(target * 1e9)
    best = None

    def collect(frame):
        nonlocal best
        stamp = int(Fraction(frame.pts) * frame.time_base * 1e9)
        if best is None or abs(stamp - target_ns) < abs(best[0] - target_ns):
            best = (
                stamp,
                cv2.remap(frame.to_ndarray(format="bgr24"), *maps, cv2.INTER_LINEAR),
            )

    with Path(entry["path"]).open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, record in reader.iter_messages(
            topics=["/image_left_raw/h265"],
            start_time=max(entry["first_record_ns"], target_ns - 8_000_000_000)
            if seek
            else None,
            end_time=target_ns + 700_000_000,
        ):
            if channel.id not in decoders:
                decoders[channel.id] = decoder_factory.decoder_for(
                    channel.message_encoding, schema
                )
            msg = decoders[channel.id](record.data)
            packet = av.Packet(bytes(msg.data))
            packet.pts = packet.dts = time_ns(msg.timestamp)
            packet.time_base = Fraction(1, 1_000_000_000)
            try:
                for frame in codec.decode(packet):
                    collect(frame)
            except av.error.FFmpegError:
                continue  # seek prefix may start before the next decodable keyframe
        try:
            for frame in codec.decode(None):
                collect(frame)
        except av.error.FFmpegError:
            pass
    if (best is None or abs(best[0] - target_ns) > 100_000_000) and seek:
        return nearest_image(entry, target, maps, seek=False)
    if best is None or abs(best[0] - target_ns) > 100_000_000:
        raise RuntimeError("no acquisition-near image for revisit")
    return best, seek


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()
    result = json.loads((args.audit / "result.json").read_text())
    entry = next(
        e for e in json.loads(args.inventory.read_text()) if e["name"] == result["bag"]
    )
    c = entry["calibration"]
    intrinsic = np.array(c["K"]).reshape(3, 3)
    output_intrinsic = intrinsic.copy()
    output_intrinsic[0] *= 640 / c["width"]
    output_intrinsic[1] *= 360 / c["height"]
    maps = cv2.initUndistortRectifyMap(
        intrinsic,
        np.array(c["D"]),
        np.eye(3),
        output_intrinsic,
        (640, 360),
        cv2.CV_32FC1,
    )
    cv2.setNumThreads(1)
    av.logging.set_level(av.logging.FATAL)
    cache = {}
    metadata = []
    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22
    )
    for event in result["events"]:
        canvas = Image.new("RGB", (1280, 470), "white")
        draw = ImageDraw.Draw(canvas)
        for side, index in [("reference", 1), ("revisit", 0)]:
            target = event["stamps"][index]
            if target not in cache:
                cache[target] = nearest_image(entry, target, maps)
            (stamp, image), seek = cache[target]
            offset = 0 if side == "reference" else 640
            canvas.paste(
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (offset, 55)
            )
            draw.text(
                (offset + 10, 10),
                f"{side}  {(stamp - entry['first_record_ns']) * 1e-9:.2f} s",
                font=font,
                fill="black",
            )
            metadata.append(
                {
                    "event": event["event"],
                    "side": side,
                    "target_stamp_s": target,
                    "image_stamp_ns": stamp,
                    "offset_ms": (stamp - target * 1e9) * 1e-6,
                    "bounded_seek_succeeded": seek,
                }
            )
        draw.text(
            (10, 430),
            "Same place may face different directions; images aid review and are not pose ground truth.",
            font=font,
            fill="black",
        )
        canvas.save(args.audit / f"event_{event['event']:02d}_images.png")
    (args.audit / "image_evidence.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
