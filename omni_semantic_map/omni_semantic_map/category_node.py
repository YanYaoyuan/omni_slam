"""ROS 2 Humble category-evidence consumer, independent of LIVO tracking.

Input is versioned evidence JSON, not raw image/COCO detections. The producer
must deliver contiguous revisions with acknowledgement; delta messages cannot
use a latest-frame dropping queue. SAM2/SigLIP live inference is a separate stage.
"""
import json
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
from std_msgs.msg import String
from .category_stream import CategoryStream


class SemanticCategories(Node):
    def __init__(self):
        super().__init__('omni_semantic_categories')
        defaults=dict(vocabulary_directory='',feature_space='',output_directory='',max_observations=20000,max_objects=5000,
                      input_topic='/omni/semantic/evidence',output_topic='/omni/semantic/category_updates')
        for k,v in defaults.items():self.declare_parameter(k,v)
        p={k:self.get_parameter(k).value for k in defaults}
        vocab=Path(p['vocabulary_directory']);self.output=Path(p['output_directory'])
        if not p['output_directory'] or not p['vocabulary_directory']:raise ValueError('explicit vocabulary and output directories required')
        if self.output.exists() and any(self.output.iterdir()):raise ValueError('new or empty output directory required')
        self.stream=CategoryStream(np.load(vocab/'text_features.npy'),json.loads((vocab/'vocabulary.json').read_text()),p['feature_space'],max_observations=p['max_observations'],max_objects=p['max_objects'])
        self.output.mkdir(parents=True,exist_ok=True);self.parameters=p
        (self.output/'run.json').write_text(json.dumps(dict(parameters=p,scope='Category-only evidence consumer. No raw image inference or geometry ownership.',started_unix=time.time()),indent=2))
        qos=QoSProfile(depth=10,reliability=ReliabilityPolicy.RELIABLE)
        self.publisher=self.create_publisher(String,p['output_topic'],qos)
        self.status=self.create_publisher(String,p['output_topic']+'/status',QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.subscription=self.create_subscription(String,p['input_topic'],self.receive,qos)
        self.rejected=0

    def receive(self,msg):
        began=time.monotonic()
        try:
            event=json.loads(msg.data);out=self.stream.apply(event)
        except (ValueError,KeyError,TypeError) as exc:
            self.rejected+=1;status=dict(ok=False,last_committed_revision=self.stream.revision,error=str(exc),rejected=self.rejected)
            self.status.publish(String(data=json.dumps(status)));self.get_logger().error(str(exc));return
        payload=json.dumps(out,allow_nan=False);temp=self.output/'categories.json.tmp';temp.write_text(payload);temp.replace(self.output/'categories.json')
        with (self.output/'updates.jsonl').open('a') as f:f.write(payload+'\n')
        self.publisher.publish(String(data=payload))
        self.status.publish(String(data=json.dumps(dict(ok=True,revision=out['revision'],objects=len(out['objects']),stored_observations=out['stored_observations'],processing_s=time.monotonic()-began,rejected=self.rejected))))


def main(args=None):
    rclpy.init(args=args);node=None
    try:
        node=SemanticCategories();rclpy.spin(node)
    finally:
        if node is not None:node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
