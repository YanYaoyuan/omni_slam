"""Explicit local weights only; model inference is outside the LIVO process."""
import hashlib
from pathlib import Path
import cv2
import numpy as np


class TrialModels:
    def __init__(self, weights, ground_model="", device="cpu", threads=2, score=.7):
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        torch.set_num_threads(threads)
        cv2.setNumThreads(1)
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        digest = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if not digest.startswith("73cbd019"):
            raise ValueError("expected official Mask R-CNN ResNet50 FPN v2 COCO_V1 weights")
        self.detector = maskrcnn_resnet50_fpn_v2(
            weights=None, weights_backbone=None, min_size=360, max_size=640)
        self.detector.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.detector.to(device).eval()
        self.ground = None
        self.provenance = {"detector": "maskrcnn_resnet50_fpn_v2_coco", "sha256": digest,
                           "device": device, "score": score, "ground_model": ground_model}
        if ground_model:
            from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
            self.processor = AutoImageProcessor.from_pretrained(ground_model, local_files_only=True)
            self.ground = SegformerForSemanticSegmentation.from_pretrained(
                ground_model, local_files_only=True).to(device).eval()
            expected = {0: "road", 1: "sidewalk", 9: "terrain"}
            if any(self.ground.config.id2label.get(k) != v for k, v in expected.items()):
                raise ValueError("ground model must use Cityscapes label IDs")
        self.device, self.score = device, score

    def __call__(self, image):
        import torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device) / 255
        labels = {1: "person", 3: "car", 11: "fire_hydrant", 64: "potted_plant"}
        detections = []; ground = np.zeros(image.shape[:2], bool)
        with torch.inference_mode():
            output = self.detector([tensor])[0]
            for i, (label, score) in enumerate(zip(output["labels"].tolist(), output["scores"].tolist())):
                if label not in labels or score < self.score or len(detections) >= 20:
                    continue
                detections.append(dict(label=labels[label], score=score,
                    box=output["boxes"][i].cpu().numpy(),
                    mask=output["masks"][i, 0].cpu().numpy() >= .6))
            if self.ground is not None:
                inputs = self.processor(images=rgb, return_tensors="pt",
                    size={"height": 384, "width": 640}).to(self.device)
                logits = self.ground(**inputs).logits
                logits = torch.nn.functional.interpolate(logits, size=image.shape[:2],
                    mode="bilinear", align_corners=False)
                confidence, classes = logits.softmax(1).max(1)
                ground = np.isin(classes[0].cpu().numpy(), [0, 1, 9]) & (confidence[0].cpu().numpy() >= .65)
        return detections, ground
