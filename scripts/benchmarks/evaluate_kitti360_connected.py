"""Same scoring protocol for the completed garage baseline transfer."""
import argparse
from pathlib import Path
from paper_final_common import read, dump, sha
import evaluate_kitti360_development as evaluate
from kitti360_connected_mapping import OUT


def configure():
    prior = evaluate.SPARSE
    evaluate.SPARSE = OUT
    evaluate.OUT = OUT/'evaluation'
    evaluate.METHODS = dict(native_indoor_history2=evaluate.NATIVE/'maps/cg/history2',
        scale_only_history2=prior/'maps/cg/history2', cg_sparse=OUT/'maps/cg/cg', history2=OUT/'maps/cg/history2',
        sor=OUT/'maps/cleanup/sor', ours_visibility1=OUT/'maps/cleanup/visibility1',
        simple_ratio=OUT/'maps/cleanup/simple_ratio', mc_point_ratio=OUT/'maps/cleanup/mc_point_ratio')


def main(action):
    configure()
    if action == 'freeze':
        evaluate.freeze()
        path = evaluate.OUT/'protocol.json'
        p = read(path)
        p['source_sha256'][str(Path(__file__).relative_to(evaluate.ROOT))] = sha(__file__)
        p['completion_note'] = 'Adds existing largest-component input filtering to garage scale adapter; no new masks, thresholds, training or invented method contribution. Earlier scale-only development scores were seen.'
        dump(path, p)
    else: evaluate.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze','run'])
    args = parser.parse_args()
    main(args.action)
