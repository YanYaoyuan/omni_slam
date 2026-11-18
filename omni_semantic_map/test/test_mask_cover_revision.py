import numpy as np
from omni_semantic_map.mask_cover_revision import explain_surface


def test_partial_masks_complete_one_surface():
    r=explain_surface([[1,1,0,0],[0,0,1,1]],np.ones(4),np.ones(4,bool),[.25,.25])
    assert r['single']['inside']==2 and r['multi']['inside']==4
    assert r['multi']['utility']==3.5 and r['multi']['queries']==[0,1]


def test_shared_outside_and_inside_are_charged_once():
    r=explain_surface([[1,1,0,1],[0,1,1,1]],np.ones(4),[1,1,1,0],[.1,.1])
    assert r['multi']['inside']==3 and r['multi']['outside']==1
    assert abs(r['multi']['utility']-1.8)<1e-12


def test_leakage_cannot_be_positive_support():
    r=explain_surface([[1,1,1,1]],np.ones(4),[1,0,0,0],[0.])
    assert r['multi']['queries']==[] and r['single']['utility']==0


def test_duplicate_queries_do_not_multiply_evidence():
    r=explain_surface([[1,1],[1,1]],np.ones(2),[1,1],[.2,.2])
    assert r['multi']['queries']==[0] and r['multi']['utility']==1.8


def test_fractional_cell_mass_and_absent_region():
    r=explain_surface([[1,1,1,0],[0,1,1,1]],[.5,.5,1.,1.],[1,1,1,1],[.1,.1])
    assert r['multi']['inside']==3 and abs(r['multi']['utility']-2.8)<1e-12
    assert explain_surface([[1,1]],[1.,1.],[0,0],[.1])['multi']['queries']==[]
