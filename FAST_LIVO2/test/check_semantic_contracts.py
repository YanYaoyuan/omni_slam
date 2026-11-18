#!/usr/bin/env python3
"""Use a captured real packet to check rejection, overload and session isolation."""
import argparse
import copy
from pathlib import Path
import tempfile
import rclpy
from rclpy.serialization import deserialize_message
from omni_slam_interfaces.msg import LivoSemanticFrame
from omni_semantic_map import live_node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packet", type=Path)
    args = ap.parse_args()
    packet = deserialize_message(args.packet.read_bytes(), LivoSemanticFrame)
    # This tests transport/lifecycle contracts, not inference; real inference has its own replay test.
    class Models:
        def __init__(self, *args):
            self.provenance = {"test": True}
    live_node.TrialModels = Models
    with tempfile.TemporaryDirectory() as temp:
        rclpy.init(args=["--ros-args", "-p", "weights:=unused", "-p", f"output_directory:={temp}"])
        node = live_node.SemanticTrial()
        try:
            wrong = copy.deepcopy(packet); wrong.image.header.stamp.sec += 1
            node.receive(wrong)
            assert node.rejected == 1 and node.slot.value is None
            node.receive(packet)
            newer = copy.deepcopy(packet); newer.keyframe.id += 1
            for stamp in [newer.keyframe.header.stamp, newer.keyframe.cloud.header.stamp,
                          newer.keyframe.image_stamp, newer.image.header.stamp, newer.camera_info.header.stamp]:
                stamp.sec += 1
            node.receive(newer)
            assert node.slot.value.keyframe.id == newer.keyframe.id and node.slot.dropped == 1
            reset = copy.deepcopy(newer); reset.keyframe.id = 0
            node.receive(reset)
            assert node.state == "session_mismatch" and node.rejected == 2
            node.tick()
            assert node.slot.value is None and node.future is None
            print("PASS: mismatched image epoch rejected; latest frame replaces backlog; session reset stops processing")
        finally:
            node.close(); node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
