"""Isolated range download for the already-frozen four-window selection."""
from run_intrinsic_public_holdout import isolated_adapter
from paper_final_common import ROOT
if __name__=='__main__':
 n=isolated_adapter('download_kitti360_windows.py');n.BASE=ROOT/'build/datasets/reviewer_extension_20261117';n.OUT=n.BASE/'data';n.main()
