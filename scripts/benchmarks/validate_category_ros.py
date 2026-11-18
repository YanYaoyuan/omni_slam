"""ROS transport test of every real LIVO cached evidence revision, plus faults."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import rclpy
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'omni_semantic_map'))
from omni_semantic_map.flat_benchmark import sha256


def main():
    dest=ROOT/'build/benchmarks/evidence_retention_20261111/livo_category_stream_v2'
    protocol=json.loads((dest/'protocol.json').read_text());summary=json.loads((dest/'summary.json').read_text());assert summary['complete']
    for p,h in protocol['sources'].items():assert sha256(p)==h
    assert sha256(dest/'events.jsonl')==summary['events_sha256'];assert sha256(dest/'expected.jsonl')==summary['expected_sha256']
    output=dest/'ros_validation';output.mkdir(exist_ok=False)
    topic=f'/omni/category_validation_{os.getpid()}';rclpy.init();node=rclpy.create_node('category_evidence_validation')
    qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
    pub=node.create_publisher(String,topic+'/evidence',qos);received={};statuses=[]
    node.create_subscription(String,topic+'/updates',lambda msg:received.update({json.loads(msg.data)['revision']:json.loads(msg.data)}),qos)
    node.create_subscription(String,topic+'/updates/status',lambda msg:statuses.append(json.loads(msg.data)),QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
    def wait(predicate,timeout=30):
        until=time.monotonic()+timeout
        while not predicate():
            if child.poll() is not None:raise RuntimeError('category node exited')
            if time.monotonic()>until:raise TimeoutError('category acknowledgement timed out')
            rclpy.spin_once(node,timeout_sec=.02)
    log=(output/'node.log').open('x')
    child=subprocess.Popen([sys.executable,'-m','omni_semantic_map.category_node','--ros-args',
        '-p','vocabulary_directory:='+protocol['vocabulary_directory'],'-p','feature_space:='+protocol['feature_space'],
        '-p','output_directory:='+str(output/'node'),'-p','input_topic:='+topic+'/evidence','-p','output_topic:='+topic+'/updates'],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result=dict(complete=False,valid_revisions=0,faults=[],max_score_error=0.,max_weight_error=0.,source_sha256=sha256(__file__))
    try:
        wait(lambda:pub.get_subscription_count()>0 and node.count_publishers(topic+'/updates')>0)
        with (dest/'events.jsonl').open() as events,(dest/'expected.jsonl').open() as expected:
            for index,(line,refline) in enumerate(zip(events,expected)):
                event=json.loads(line);ref=json.loads(refline)
                if index in [0,1]:
                    faults=[]
                    if index==0:
                        wrong=deepcopy(event);wrong['feature_space']='not_siglip';faults.append(('wrong_feature_space',wrong))
                    else:
                        wrong=deepcopy(event);wrong['revision']=0;faults.append(('duplicate_revision',wrong))
                        wrong=deepcopy(event);wrong['revision']=2;faults.append(('missing_revision',wrong))
                        wrong=deepcopy(event);wrong['session_id']='different_run';faults.append(('mixed_session',wrong))
                        wrong=deepcopy(event);wrong['objects'][0]['observation_ids'].append(999999);faults.append(('unseen_observation',wrong))
                    for label,bad in faults:
                        before=len(statuses);pub.publish(String(data=json.dumps(bad)));wait(lambda:len(statuses)>before and not statuses[-1]['ok'])
                        assert statuses[-1]['last_committed_revision']==index-1
                        result['faults'].append(dict(case=label,status=statuses[-1]))
                pub.publish(String(data=line));wait(lambda:index in received,timeout=60);actual=received.pop(index)
                assert actual.keys()==ref.keys()
                for k in actual.keys()-{'objects'}:assert actual[k]==ref[k],k
                assert len(actual['objects'])==len(ref['objects'])
                for a,b in zip(actual['objects'],ref['objects']):
                    for k in ['id','retained','geometry_revision']:assert a[k]==b[k]
                    for k in ['name','replica_class_id','selected_observation_ids','fallback','method']:assert a['category'][k]==b['category'][k]
                    for k in ['score','margin','weights']:np.testing.assert_allclose(a['category'][k],b['category'][k],rtol=0,atol=1e-10)
                    result['max_score_error']=max(result['max_score_error'],abs(a['category']['score']-b['category']['score']))
                    result['max_weight_error']=max(result['max_weight_error'],float(np.max(np.abs(np.array(a['category']['weights'])-b['category']['weights']))))
                result['valid_revisions']+=1
                if index%50==0:print('ROS VALIDATED',index,flush=True)
        assert result['valid_revisions']==summary['revisions']
        result.update(complete=True,transport='ROS2 Humble reliable String evidence / category messages, one event then acknowledgement',
            no_inference_claim=True,semantics='No publisher for live SAM2/SigLIP inference yet; this validates cached real LIVO evidence through the actual ROS consumer.')
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=10)
        log.close();node.destroy_node();rclpy.shutdown();(output/'summary.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE ROS VALIDATION',result['valid_revisions'],len(result['faults']),flush=True)


if __name__=='__main__':main()
