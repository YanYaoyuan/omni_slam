"""Atomic category updates over externally owned, keyframe-anchored objects.

Events contain new immutable SigLIP evidence and a complete membership snapshot.
This module cannot create/delete geometry or use new labels for association.
Contiguous revisions detect lost delta packets; restart requires a new stream.
"""
import numpy as np
from .consensus_category import classify


class CategoryStream:
    def __init__(self,text,vocabulary,feature_space,*,max_observations=20000,max_objects=5000):
        self.text=np.asarray(text,np.float32)
        if self.text.ndim!=2 or len(self.text)<2 or not np.isfinite(self.text).all() or (np.linalg.norm(self.text,axis=1)<=1e-12).any():
            raise ValueError('invalid text features')
        if len(vocabulary['names'])!=len(self.text) or len(vocabulary['replica_class_ids'])!=len(self.text):raise ValueError('vocabulary dimensions disagree')
        self.text=self.text/np.linalg.norm(self.text,axis=1,keepdims=True)
        if not isinstance(feature_space,str) or not feature_space:raise ValueError('explicit feature-space fingerprint required')
        if max_observations<1 or max_objects<1:raise ValueError('positive capacity required')
        self.vocab=vocabulary;self.feature_space=feature_space
        self.max_observations=max_observations;self.max_objects=max_objects
        self.observations={};self.session=None;self.revision=-1;self.frame=-1

    def apply(self,event):
        if not isinstance(event,dict):raise ValueError('event must be an object')
        if event.get('schema_version')!=1 or event.get('feature_space')!=self.feature_space:raise ValueError('feature schema/space mismatch')
        session=event.get('session_id');revision=event.get('revision');frame=event.get('frame_id')
        if not isinstance(session,str) or not session or (self.session is not None and session!=self.session):raise ValueError('session mismatch; start a new stream')
        if type(revision) is not int or revision!=self.revision+1:raise ValueError('noncontiguous revision; delta packet missing or duplicated')
        if type(frame) is not int or frame<self.frame:raise ValueError('out-of-order keyframe')
        incoming=event.get('observations',[]);objects=event.get('objects',[])
        if not isinstance(incoming,list) or not isinstance(objects,list):raise ValueError('observation and object arrays required')
        if len(self.observations)+len(incoming)>self.max_observations or len(objects)>self.max_objects:raise ValueError('capacity exceeded; no eviction of referenced evidence')
        additions={}
        for r in incoming:
            if not isinstance(r,dict):raise ValueError('invalid observation record')
            oid=r.get('id');fid=r.get('frame_id');area=r.get('area')
            if type(oid) is not int or oid<0 or oid in self.observations or oid in additions:raise ValueError('immutable observation ID reused')
            if type(fid) is not int or fid!=frame:raise ValueError('new evidence must belong to current keyframe')
            if type(area) is not int or area<=0:raise ValueError('invalid mask area')
            feature=np.asarray(r['feature'],np.float32)
            if feature.shape!=(self.text.shape[1],) or not np.isfinite(feature).all() or np.linalg.norm(feature)<=1e-12:raise ValueError('invalid SigLIP descriptor')
            additions[oid]=dict(frame_id=fid,area=area,feature=feature,evidence=r.get('evidence'))
        available={**self.observations,**additions};owners=set();object_ids=set();updates=[]
        for obj in objects:
            if not isinstance(obj,dict):raise ValueError('invalid object record')
            oid=obj.get('id');members=obj.get('observation_ids');retained=obj.get('retained')
            if not isinstance(oid,str) or not oid or oid in object_ids:raise ValueError('duplicate or invalid object ID')
            if type(retained) is not bool or not isinstance(obj.get('geometry_revision'),str):raise ValueError('upstream retention and geometry revision required')
            if not isinstance(members,list) or not members or any(type(i) is not int for i in members) or len(set(members))!=len(members):raise ValueError('invalid membership')
            if not set(members)<=available.keys():raise ValueError('unseen observation referenced')
            if owners.intersection(members):raise ValueError('observation shared across objects')
            owners.update(members);object_ids.add(oid)
            # Local array indices must preserve global observation-ID ties.
            members=sorted(members)
            values=[available[i] for i in members]
            result=classify(list(range(len(values))),np.array([v['feature'] for v in values]),np.array([v['frame_id'] for v in values]),
                np.array([v['area'] for v in values]),self.text,self.vocab)
            selected=[members[i] for i in result['selected_observation_ids']];result['selected_observation_ids']=selected
            best=sorted(range(len(selected)),key=lambda i:(-result['weights'][i],selected[i]))[:3]
            updates.append(dict(id=oid,retained=retained,geometry_revision=obj['geometry_revision'],category=result,
                evidence=[dict(observation_id=selected[i],reference=available[selected[i]]['evidence'],weight=result['weights'][i]) for i in best]))
        # Commit only after the complete event has been validated and evaluated.
        self.observations.update(additions);self.session=session;self.revision=revision;self.frame=frame
        return dict(schema_version=1,session_id=session,revision=revision,frame_id=frame,feature_space=self.feature_space,
            objects=updates,stored_observations=len(self.observations),category_only=True)
