"""Real reference geometry and every score from the identity audit."""
import html
from pathlib import Path
import numpy as np
from paper_final_common import ROOT, read, dump, sha
from evaluate_kitti360_development import read_static_ply
from audit_reference_identity import OUT, ARMS, verify

DEST = ROOT/'output/reference_identity_audit_v1'
NAMES = {'baseline':'Frozen strong baseline', 'point':'Point loss',
         'object_balanced':'Object-balanced loss', 'mean_neighbor':'Mean-neighbor map loss',
         'ordered_map':'Ours candidate / ordered map credit'}
MODE_NAMES = {'upstream_train':'Published-reader-compatible primary replay',
              'raw_identity':'Diagnostic: split aliases, retain all raw targets',
              'target_only':'Diagnostic: raw building/car, aliases become void'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = verify(); s = read(OUT/'summary.json')
    assert s['complete'] and not (OUT/'report_audit.json').exists()
    for f, h in s['files'].items():
        assert sha(f) == h, f
    DEST.mkdir(parents=True, exist_ok=False)
    figures, sections, chosen = [], [], {}
    for c in p['contexts']:
        key = c['key']; r = s['results'][key]
        groups = read(OUT/key/'alias_groups.json')['collisions']
        eligible = [g for g in groups if g['eligible']]
        selected = max(eligible, key=lambda g:(max(x['minimum_distance_m'] for x in g['separations']), -g['upstream_id']))
        chosen[key] = dict(upstream_id=selected['upstream_id'],
                           rule='Largest exact minimum distance between raw constituents; GT diagnostic, not a method-success selection')
        raw = read_static_ply(c['raw_reference']); ref = dict(np.load(c['reference']))
        fig, axes = plt.subplots(1, 2, figsize=(15,6), constrained_layout=True)
        sample = np.arange(0, len(ref['xyz']), max(1,len(ref['xyz'])//90000))
        axes[0].scatter(ref['xyz'][sample,0], ref['xyz'][sample,1], s=.25, color='#dde2e5', rasterized=True)
        for color, part in zip(['#007e86','#e45255','#a076c4'], selected['constituents']):
            idx = np.flatnonzero((raw['semantic']==part['raw_semantic']) & (raw['instance']==part['raw_instance']))
            idx = idx[::max(1,len(idx)//30000)]
            axes[0].scatter(ref['xyz'][idx,0], ref['xyz'][idx,1], s=1.5, color=color,
                            label=f"Raw {part['raw_name']} {part['raw_instance']}", rasterized=True)
        separation = max(x['minimum_distance_m'] for x in selected['separations'])
        axes[0].set_title(f"Both become reference ID {selected['upstream_id']}\nMinimum point-to-point separation: {separation:.2f} m")
        axes[0].set_aspect('equal'); axes[0].set_xlabel('World x / m'); axes[0].set_ylabel('World y / m')
        axes[0].legend(loc='best', markerscale=4, fontsize=9)
        rows = [o for o in r['modes']['target_only']['objects'] if o['gt_id']//1000==2]
        xx = np.arange(len(rows))
        axes[1].bar(xx-.24, [v['methods']['baseline']['best_iou'] for v in rows], width=.24,
                    label='Actual baseline best IoU', color='#526e89')
        axes[1].bar(xx, [v['coverage']['legacy_camera_sources'] for v in rows], width=.24,
                    label='Camera-source coverage', color='#e6ae4d')
        axes[1].bar(xx+.24, [v['coverage']['full_measured_lidar'] for v in rows], width=.24,
                    label='Full measured LiDAR coverage', color='#00a38b')
        axes[1].axhline(.5, ls='--', lw=1, color='#555555')
        axes[1].set_xticks(xx, [str(v['gt_id']) for v in rows], rotation=90)
        axes[1].set_ylim(0,1.02); axes[1].set_ylabel('Confidence-weighted fraction')
        axes[1].set_xlabel('Every eligible raw building, garage aliases excluded')
        axes[1].set_title('Real building formation still has missing support\nCoverage is a necessary condition, not achievable IoU or AP')
        axes[1].legend(fontsize=8, loc='upper right')
        fig.suptitle(key+' / actual reference geometry / diagnostic only / no new predictions')
        f = DEST/(key+'.png'); fig.savefig(f,dpi=160); plt.close(fig); figures.append(str(f))
        collision_rows = ''.join('<tr><td>'+str(g['upstream_id'])+'</td><td>'+
            ', '.join(str(v['raw_instance']) for v in g['constituents'])+'</td><td>'+
            f"{max(x['minimum_distance_m'] for x in g['separations']):.2f}"+'</td></tr>' for g in eligible)
        sections.append(f'<h2>{key}</h2><img src="{key}.png"><details><summary>All {len(eligible)} eligible identity collisions</summary>'+
                        '<table><tr><th>Folded ID</th><th>Distinct raw annotation IDs</th><th>Min separation (m)</th></tr>'+collision_rows+'</table></details>')
        for mode, mr in r['modes'].items():
            score_rows = ''.join('<tr><td>'+NAMES[m]+'</td>'+''.join(f'<td>{mr["methods"][m]["metrics"][metric]*100:.6f}</td>' for metric in ['allAp','allAp50%'])+
                '<td>'+html.escape(str(mr['methods'][m]['versus_baseline']))+'</td></tr>' for m in ARMS)
            object_rows = ''.join('<tr><td>'+str(o['gt_id'])+'</td><td>'+', '.join(str(v['raw_instance']) for v in o['raw_constituents'])+'</td>'+
                ''.join(f'<td>{o["methods"][m]["best_iou"]:.6f}</td>' for m in ARMS)+f'<td>{o["fixed_pool_upper_iou"]:.6f}</td></tr>' for o in mr['objects'])
            sections.append('<h3>'+MODE_NAMES[mode]+'</h3><p>Eligible GT: '+str(mr['eligible_gt'])+
                '. Old candidate pool potential new AP50 matches: '+str(mr['potential_new_ap50_in_old_pool'])+'</p>'+
                '<table><tr><th>Unchanged prediction</th><th>AP50:90 (%)</th><th>AP50 (%)</th><th>vs same-mode baseline</th></tr>'+score_rows+'</table>'+
                '<details><summary>Every eligible object: best same-class IoU, distinct from AP matching</summary><table><tr><th>Encoded GT</th><th>Raw IDs</th>'+
                ''.join('<th>'+NAMES[m]+'</th>' for m in ARMS)+'<th>Optimistic fixed-pool bound</th></tr>'+object_rows+'</table></details>')
    table = ''
    for mode, scores in s['means'].items():
        table += '<tr><td>'+MODE_NAMES[mode]+'</td>'+''.join(f'<td>{scores[m]*100:.6f}</td>' for m in ARMS)+'</tr>'
    text = '''<!doctype html><meta charset="utf-8"><title>omni_slam reference identity audit</title>
<style>body{max-width:1400px;margin:40px auto;padding:0 20px;font:16px/1.5 sans-serif;color:#163343}img{width:100%}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{padding:8px;border-bottom:1px solid #d5dfe3;text-align:right}td:first-child,th:first-child{text-align:left}details{margin:18px 0;overflow:auto}.note{background:#fff3d8;padding:15px}</style>
<h1>Reference identity audit — unchanged omni_slam predictions</h1>
<p class="note">30 complete score replays, zero new maps/training. Only the first row follows the frozen public-reader-compatible protocol. The other rows are explicitly alternative reference diagnostics. A score change across rows is NOT an algorithm improvement or an official leaderboard score.</p>
<p>The public reader folds building and garage into train class 2, then takes the raw instance suffix modulo 1000. Different raw annotation identities can therefore become one evaluated object. This audit preserves original primary scores and all predictions, and separates identity folding from ignoring the garage alias.</p>
<table><tr><th>Reference definition</th>'''+''.join('<th>'+NAMES[m]+'</th>' for m in ARMS)+'</tr>'+table+'</table>'+''.join(sections)+'''
<p>Source: <a href="https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/evaluation/semantic_3d/evalInstanceLevelSemanticLabeling.py">KITTI-360 public reference reader</a>; <a href="https://raw.githubusercontent.com/autonomousvision/kitti360Scripts/master/kitti360scripts/helpers/labels.py">raw taxonomy</a>. Server behavior is not established by this local audit. No claim that the complete method goal, robot validation or independent assessment is achieved.</p>'''
    (DEST/'index.html').write_text(text)
    dump(OUT/'report_audit.json', dict(complete=True, figures=figures, selection=chosen,
        all_results_rendered=True, source_sha256=sha(__file__),
        files={str(f):sha(f) for f in DEST.iterdir()}, summary_sha256=sha(OUT/'summary.json')))
    print('REFERENCE REPORT', DEST/'index.html', flush=True)


if __name__ == '__main__':
    main()
