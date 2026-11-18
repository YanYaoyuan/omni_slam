"""Run full-duration LIVO strong semantics after a completed input conversion.

No training or benchmark retuning. Own subprocesses only; no process replacement
or overwrite of prior runs. The 30-second diagnostic remains separate.
"""
import json
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import struct
import numpy as np
import yaml
from audit_evidence_retention import ROOT, OUT, dump
from omni_semantic_map.flat_benchmark import sha256


def camera_age_preflight(bag,max_age):
    """Read the initial 10s acquisition epochs, without decoding image pixels."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image,PointCloud2
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='mcap'),rosbag2_py.ConverterOptions('',''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/experiment/image','/experiment/lidar']))
    times={'/experiment/image':[],'/experiment/lidar':[]};start=None
    while reader.has_next():
        topic,payload,record_stamp=reader.read_next()
        if start is None:start=record_stamp
        if record_stamp>start+10_000_000_000:break
        assert payload[:2]==b'\x00\x01','requires validated little-endian CDR'
        sec,nsec=struct.unpack_from('<iI',payload,4);stamp=sec*10**9+nsec
        if not times[topic]:
            msg=deserialize_message(payload,Image if topic.endswith('image') else PointCloud2)
            assert stamp==msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
        times[topic].append(stamp)
    image=np.array(times['/experiment/image'],dtype=np.int64);lidar=np.array(times['/experiment/lidar'],dtype=np.int64)
    assert len(image)>20 and len(lidar)>20
    index=np.searchsorted(image,lidar,side='right')-1;valid=index>=0
    ages=(lidar[valid]-image[index[valid]])/1e9
    result=dict(sample_seconds=10,image_messages=len(image),lidar_messages=len(lidar),
        median_image_age_ms=float(np.median(ages)*1000),p95_image_age_ms=float(np.percentile(ages,95)*1000),
        within_configured_gate_fraction=float(np.mean(ages<=max_age)),configured_max_age_s=max_age,
        scope='Source header epochs, before estimator scheduling; validates temporal phase, not executed visual updates.')
    assert result['within_configured_gate_fraction']>=.9,result
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    assert args.tag.replace('_','').isalnum(),'invalid output tag'
    prepared=args.input;audit=json.loads((prepared/'input_audit.json').read_text())
    original=yaml.safe_load(Path('/home/user/vbot/data/rosbag2245/metadata.yaml').read_text())['rosbag2_bagfile_information']
    config=ROOT/'build/experiments/semantic_20261106/managed_config/livo.yaml';params=yaml.safe_load(config.read_text())['/**']['ros__parameters']
    K=np.array([[params['camera']['fx'],0,params['camera']['cx']],[0,params['camera']['fy'],params['camera']['cy']],[0,0,1.]])
    np.testing.assert_allclose(K,audit['camera']['output_K'],atol=1e-8)
    Tcl=np.eye(4);Tcl[:3,:3]=np.array(params['extrin_calib']['Rcl']).reshape(3,3);Tcl[:3,3]=params['extrin_calib']['Pcl']
    np.testing.assert_allclose(Tcl,audit['T_camera_lidar'],atol=1e-8)
    duration=original['duration']['nanoseconds']/1e9
    assert audit['duration_requested_s']>=duration and not audit['decode_errors']
    assert audit['streams']['image']['rate_hz']>=29,'full visual input required; decimation can violate the image-age gate'
    spans={k:(v['last_ns']-v['first_ns'])/1e9 for k,v in audit['streams'].items()}
    assert all(x>=duration-2 for x in spans.values()) and all(v['nonmonotonic']==0 for v in audit['streams'].values())
    timing=camera_age_preflight(prepared/'bag',params['vio']['max_image_age_s'])
    status_path=OUT/f'livo_full_validation_{args.tag}_status.json';assert not status_path.exists()
    state=dict(complete=False,pid=os.getpid(),phase='starting',source_sha256=sha256(Path(__file__)),steps=[],
        full_source_duration_s=duration,input_audit_sha256=sha256(prepared/'input_audit.json'),input_spans_s=spans,camera_age_preflight=timing,
        input_sampling='Decode all H265 packets and retain every timestamped RGB frame (~30Hz), all selected lidar/IMU over full recording.',
        scope='Full temporal recording coverage, staged semantic processing after native LIVO capture. Not online end-to-end latency.')
    def publish():state['updated_epoch']=time.time();dump(status_path,state)
    publish();env=os.environ.copy();env.update(ROS_DOMAIN_ID='95',ROS_LOCALHOST_ONLY='0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HOME=str(ROOT/'build/benchmarks/ovo_full_20261110/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLCONFIGDIR='/tmp/omni-retention-mpl',PYTHONUNBUFFERED='1')
    py=ROOT/'build/benchmarks/ovo_full_20261110/venv311/bin/python';cgpy=ROOT/'build/benchmarks/expanded_semantic_20261109/cg_venv/bin/python'
    capture=OUT/f'livo_full_capture_{args.tag}';output=OUT/f'livo_full_strong_{args.tag}';job=OUT/'jobs'/f'livo_full_{args.tag}.json';scripts=ROOT/'scripts/benchmarks'
    def execute(phase,command,ros=False):
        log=OUT/'logs'/('livo_full_'+args.tag+'_'+phase+'.log');start=time.monotonic();peak=0;childenv=env.copy()
        if not ros:childenv.pop('PYTHONPATH',None)
        with log.open('x') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=childenv,stdout=stream,stderr=subprocess.STDOUT)
            state.update(phase=phase,child_pid=child.pid,log=str(log));publish()
            while child.poll() is None:
                try:
                    rows=Path(f'/proc/{child.pid}/status').read_text().splitlines()
                    rss=next(int(r.split()[1])*1024 for r in rows if r.startswith('VmRSS:'));peak=max(peak,rss)
                except (FileNotFoundError,StopIteration,ProcessLookupError):pass
                state['current_step_sampled_rss_bytes']=peak;publish();time.sleep(1)
            rc=child.returncode
        state['steps'].append(dict(phase=phase,returncode=rc,wall_s=time.monotonic()-start,sampled_peak_rss_bytes=peak,log_sha256=sha256(log),
            rss_scope='This subprocess only; nested GPU-worker RSS is logged by its own queue or remains unavailable.'))
        state.update(child_pid=None);publish()
        if rc:raise RuntimeError(f'{phase} failed with {rc}')
        print('COMPLETED',phase,flush=True)
    try:
        execute('capture',[sys.executable,scripts/'capture_livo_strong.py','--bag',prepared/'bag','--config',config,'--output',capture,'--rate','1'],ros=True)
        saved=json.loads((capture/'capture.json').read_text());stamp=[r['stamp_ns'] for r in saved['frames']]
        state['captured_span_s']=(max(stamp)-min(stamp))/1e9
        state['captured_frames']=len(stamp);state['capture_temporal_coverage']=state['captured_span_s']/duration
        assert state['capture_temporal_coverage']>=.95,'insufficient full recording visual coverage'
        execute('prepare_models',[py,scripts/'livo_strong_cache.py','--capture',capture/'capture.json','--output',output,'--job',job,'--scene','rosbag2245_full'])
        execute('frontend',[py,scripts/'run_split_queue.py','--jobs',job,'--gpu','0','--status',OUT/f'livo_full_gpu_queue_{args.tag}_status.json'])
        execute('geometry',[py,scripts/'livo_strong_cache.py','--job',job,'--geometry'])
        execute('backend',[cgpy,scripts/'run_livo_strong_backend.py','--cache',output/'cache','--output',output/'cg'])
        execute('sparse_support',[py,scripts/'filter_livo_sparse_support.py','--cache',output/'cache','--output',output/'cache_supportcc',
            '--livo-config',config,'--cg-config',ROOT/'build/benchmarks/expanded_semantic_20261109/conceptgraphs_runs/office_0/mapping_config.json'])
        execute('backend_sparse',[cgpy,scripts/'run_livo_strong_backend.py','--cache',output/'cache_supportcc','--output',output/'cg_supportcc'])
        execute('report',[py,scripts/'report_livo_strong.py','--job',job,'--report-name',f'livo_full_{args.tag}'])
        execute('report_sparse',[py,scripts/'report_livo_strong.py','--job',job,'--cache-name','cache_supportcc','--run-name','cg_supportcc','--report-name',f'livo_full_{args.tag}_supportcc'])
        state.update(complete=True,phase='complete',children_peak_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            acceptance='Execution complete; visual/query and failure audit still required before goal completion.');publish()
    except Exception as e:state.update(phase='failed',error=repr(e));publish();raise


if __name__=='__main__':main()
