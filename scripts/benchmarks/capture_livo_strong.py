"""Replay native LIVO and retain atomic inputs for the strong semantic backend.

ROS stays in Python 3.10; model workers use their isolated Python 3.11 runtime.
Every retained RGB/cloud/pose/calibration comes from one LivoSemanticFrame.
No depth synthesis, camera calibration guessing or semantic pose feedback.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy
from rclpy.serialization import serialize_message
from cv_bridge import CvBridge
from sensor_msgs_py.point_cloud2 import read_points
from omni_slam_interfaces.msg import LivoSemanticFrame, LivoKeyframe

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'omni_semantic_map'))
from omni_semantic_map.live_core import rigid_matrix
from omni_semantic_map.flat_benchmark import sha256


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
def nanoseconds(stamp):return int(stamp.sec)*10**9+int(stamp.nanosec)
def matrix(value):
    p=value.position if hasattr(value,'position') else value.translation
    q=value.orientation if hasattr(value,'orientation') else value.rotation
    return rigid_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])


def main():
    p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rate',type=float,default=.5);args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ['rgb','clouds','packets']:(output/name).mkdir()
    bag=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information'];duration=bag['duration']['nanoseconds']/1e9
    assert args.rate>0
    report=dict(complete=False,scope=__doc__,bag=str(args.bag.resolve()),playback_rate=args.rate,source_duration_s=duration,
        frames=[],rejections=[],native_keyframes=[],transport_environment={k:os.environ.get(k) for k in ['RMW_IMPLEMENTATION','ROS_LOCALHOST_ONLY','ROS_DOMAIN_ID','FASTDDS_BUILTIN_TRANSPORTS']},source_sha256=sha256(Path(__file__)),config_sha256=sha256(args.config),
        bag_sha256={str(q):sha256(q) for q in [args.bag/'metadata.yaml',*[args.bag/f for f in bag['relative_file_paths']]]})
    logs=[];processes=[];bridge=CvBridge();started=time.monotonic();last_received=None;received=0
    rclpy.init(args=[]);node=rclpy.create_node('omni_strong_semantic_capture')
    def start(command,name):
        log=(output/(name+'.log')).open('w');logs.append(log)
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append(process);return process
    def receive(message):
        nonlocal last_received,received
        received+=1;last_received=time.monotonic();k=message.keyframe
        try:
            stamp=nanoseconds(k.header.stamp);image_stamp=nanoseconds(message.image.header.stamp)
            assert k.visual_update and -10**6<=stamp-image_stamp<=10**8,'image age'
            assert k.image_stamp==message.image.header.stamp and message.camera_info.header==message.image.header,'camera/image epoch'
            assert k.cloud.header.stamp==k.header.stamp and k.cloud.header.frame_id==k.tracking_frame,'cloud epoch/frame'
            assert k.header.frame_id and k.tracking_frame and message.image.header.frame_id,'empty frame'
            if report['frames']:
                old=report['frames'][-1]
                assert int(k.id)>old['frame_id'] and stamp>old['stamp_ns'] and k.header.frame_id==old['world_frame'],'session/order'
            K=np.asarray(message.camera_info.k).reshape(3,3)
            assert np.isfinite(K).all() and K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1]),'intrinsics'
            assert np.all(np.asarray(message.camera_info.d)==0),'image must be rectified'
            assert message.image.width==message.camera_info.width and message.image.height==message.camera_info.height,'image dimensions'
            Twt=matrix(k.pose);Tct=matrix(message.camera_from_tracking);Twc=Twt@np.linalg.inv(Tct)
            rgb=bridge.imgmsg_to_cv2(message.image,'rgb8')
            cloud=read_points(k.cloud,field_names=('x','y','z'),skip_nans=True)
            if isinstance(cloud,np.ndarray) and cloud.dtype.names:points=np.column_stack([cloud[n].ravel() for n in ('x','y','z')])
            else:points=np.asarray(list(cloud),np.float32).reshape(-1,3)
            points=points[np.isfinite(points).all(1)].astype(np.float32)
            assert len(points)>0
            fid=int(k.id);image_path=output/'rgb'/f'{fid:06d}.png';cloud_path=output/'clouds'/f'{fid:06d}.npz';packet_path=output/'packets'/f'{fid:06d}.cdr'
            assert not image_path.exists() and not cloud_path.exists()
            assert cv2.imwrite(str(image_path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            np.savez_compressed(cloud_path,points=points);packet_path.write_bytes(serialize_message(message))
            report['frames'].append(dict(frame_id=fid,stamp_ns=stamp,image_stamp_ns=image_stamp,world_frame=k.header.frame_id,
                tracking_frame=k.tracking_frame,camera_frame=message.image.header.frame_id,rgb=str(image_path),cloud=str(cloud_path),
                packet=str(packet_path),pose=Twt.tolist(),camera_from_tracking=Tct.tolist(),camera_pose=Twc.tolist(),intrinsic=K.tolist(),
                width=int(message.image.width),height=int(message.image.height),points=len(points),semantic=True,
                capture_elapsed_s=last_received-started,serialization_wall_s=time.monotonic()-last_received,
                sha256={str(q):sha256(q) for q in [image_path,cloud_path,packet_path]}))
        except Exception as e:report['rejections'].append(dict(frame_id=int(k.id),reason=repr(e)))
        dump(output/'capture.json',report)
    node.create_subscription(LivoSemanticFrame,'/omni/slam/livo/semantic_frames',receive,QoSProfile(depth=8,reliability=ReliabilityPolicy.BEST_EFFORT))
    def keyframe(message):
        report['native_keyframes'].append(dict(frame_id=int(message.id),visual_update=bool(message.visual_update),stamp_ns=nanoseconds(message.header.stamp)))
    node.create_subscription(LivoKeyframe,'/omni/slam/livo/keyframes',keyframe,QoSProfile(depth=100,reliability=ReliabilityPolicy.RELIABLE))
    def spin(seconds,until=None):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.05)
            if until and until():break
    try:
        livo=start(['ros2','launch','fast_livo','managed_mapping.launch.py',f'config:={args.config.resolve()}',
            f'output_directory:={output/"livo"}',f'map_path:={output/"map.pcd"}','use_sim_time:=true'],'livo')
        spin(15,lambda:node.count_publishers('/omni/slam/livo/semantic_frames')>0 or livo.poll() is not None)
        assert livo.poll() is None and node.count_publishers('/omni/slam/livo/semantic_frames')>0,'LIVO publisher discovery failed'
        spin(2)  # Allow peer subscription discovery before the first bag frame.
        player=start(['ros2','bag','play',str(args.bag.resolve()),'--clock','50','-r',str(args.rate),'--disable-keyboard-controls','--remap',
            '/experiment/lidar:=/omni/sensors/lidar/points','/experiment/imu:=/omni/sensors/imu/data','/experiment/image:=/omni/sensors/camera/image_rect'],'player')
        spin(duration/args.rate+30,lambda:player.poll() is not None or livo.poll() is not None)
        assert player.poll()==0,'bag replay failed or timed out';spin(8);assert livo.poll() is None,'LIVO exited unexpectedly'
        assert len(report['frames'])>=5 and not report['rejections'], 'capture incomplete or rejected atomic packets'
        report.update(complete=True,received_packets=received,captured_packets=len(report['frames']),wall_s=time.monotonic()-started,
            recorder_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            note='Actual ROS LIVO replay input capture. Semantic model processing is a subsequent stage; this is not a completed end-to-end semantic result.')
    except Exception as e:report['error']=repr(e);raise
    finally:
        report['received_packets']=received
        report['captured_packets']=len(report['frames'])
        report['visual_keyframes']=sum(k['visual_update'] for k in report['native_keyframes'])
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGINT)
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
        for log in logs:log.close()
        report['children_max_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
        dump(output/'capture.json',report);node.destroy_node();rclpy.shutdown()
    print('CAPTURED',report['captured_packets'],'atomic frames',flush=True)


if __name__=='__main__':main()
