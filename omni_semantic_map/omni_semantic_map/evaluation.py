"""Identity audit for a frozen set of depth-supported object observations.

This measures association conditional on available observations, NOT detection
recall, tracking IDF1, physical-object count, or 3D localization accuracy.
Annotations must be independently reviewed; missing labels are never negatives.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def records(document):
    result = []
    for obj in document["objects"]:
        for index, observation in enumerate(obj["observations"]):
            result.append(dict(observation_id=f"{obj['id']}:{index}",
                               predicted_object_id=obj["id"],
                               predicted_label=obj["label"],
                               keyframe_id=observation["keyframe_id"],
                               stamp=observation["stamp"],
                               box=observation.get("box"), evidence=observation["evidence"],
                               valid_detection=None, gt_instance_id=None))
    ids = [r["observation_id"] for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate object/observation identity")
    return result


def annotation_template(document):
    return dict(schema_version=1, source_sha256=fingerprint(document),
                scope="Depth-supported observations only; not detector recall",
                instructions="Review all evidence independently. Set valid_detection to true/false. "
                "For true detections assign a physical-instance ID consistently across keyframes; "
                "leave uncertain entries null. Do not copy predicted_object_id as truth.",
                observations=records(document))


def evaluate(document, annotations):
    if annotations["source_sha256"] != fingerprint(document):
        raise ValueError("annotation source mismatch; do not apply labels to another map")
    predicted = {r["observation_id"]: r for r in records(document)}
    seen = set()
    contingency = Counter()
    predicted_counts = Counter()
    truth_counts = Counter()
    truth_to_pred = defaultdict(set)
    pred_to_truth = defaultdict(set)
    rejected = pending = 0
    for row in annotations["observations"]:
        key = row["observation_id"]
        if key in seen or key not in predicted:
            raise ValueError("duplicate or unknown observation annotation")
        seen.add(key)
        valid, gt = row.get("valid_detection"), row.get("gt_instance_id")
        if valid is not None and type(valid) is not bool:
            raise ValueError("valid_detection must be true, false or null")
        if valid is not True:
            if gt is not None:
                raise ValueError("instance ID requires a valid detection")
            rejected += valid is False
            pending += valid is None
            continue
        if gt is None:
            pending += 1
            continue
        if not isinstance(gt, str) or not gt.strip():
            raise ValueError("instance IDs must be nonempty strings")
        pred = predicted[key]["predicted_object_id"]
        contingency[pred, gt] += 1
        predicted_counts[pred] += 1
        truth_counts[gt] += 1
        truth_to_pred[gt].add(pred)
        pred_to_truth[pred].add(gt)
    pending += len(predicted) - len(seen)
    choose2 = lambda n: n * (n - 1) // 2
    tp = sum(choose2(n) for n in contingency.values())
    pred_pairs = sum(choose2(n) for n in predicted_counts.values())
    true_pairs = sum(choose2(n) for n in truth_counts.values())
    count = sum(truth_counts.values())
    precision = tp / pred_pairs if pred_pairs else None
    recall = tp / true_pairs if true_pairs else None
    f1 = 2 * tp / (pred_pairs + true_pairs) if pred_pairs + true_pairs else None
    return dict(scope="Association conditional on independently labeled depth-supported detections",
                status="no_identity_labels" if not count else
                ("partial_annotation" if pending else "fully_annotated_supported_observations"),
                total_observations=len(predicted), labeled_valid_observations=count,
                rejected_detection_observations=rejected, pending_observations=pending,
                identity_label_fraction=count / len(predicted) if predicted else 0.,
                true_positive_pairs=tp, predicted_same_object_pairs=pred_pairs,
                ground_truth_same_object_pairs=true_pairs,
                pair_precision=precision, pair_recall=recall, pair_f1=f1,
                split_gt_instances=sum(len(v) > 1 for v in truth_to_pred.values()),
                merged_predicted_objects=sum(len(v) > 1 for v in pred_to_truth.values()),
                fragments_per_labeled_instance={k: len(v) for k, v in sorted(truth_to_pred.items())},
                source_sha256=fingerprint(document))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["template", "evaluate"])
    ap.add_argument("objects", type=Path)
    ap.add_argument("--annotations", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    document = json.loads(args.objects.read_text())
    if args.mode == "template":
        output = annotation_template(document)
    else:
        if args.annotations is None:
            ap.error("evaluate requires --annotations")
        output = evaluate(document, json.loads(args.annotations.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
