"""One-time editorial merge of frozen V1 and current object-map evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'docs/paper/visibility_revision/latex'
NEW = ROOT / 'docs/paper/evidence_authorized_revision/latex'
ARCHIVE = ROOT / 'docs/paper/archives/visibility_revision_before_merge_2026-11-16/latex'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert ARCHIVE.is_dir() and not (OLD / 'merged_version.json').exists()
    inputs = {}
    for path in ARCHIVE.rglob('*'):
        if path.is_file():
            assert sha(path) == sha(OLD / path.relative_to(ARCHIVE)), path
    for tree in ['figures', 'tables', 'data']:
        dest = OLD / tree / 'current'
        shutil.copytree(NEW / tree, dest)
        for path in (NEW / tree).rglob('*'):
            if path.is_file(): inputs[str(path.relative_to(ROOT))] = sha(path)
    (OLD / 'tables/v1').mkdir()
    for path in (ARCHIVE / 'tables').glob('*.tex'):
        text = path.read_text().replace('Ours', 'V1')
        (OLD / 'tables/v1' / path.name).write_text(text)
        inputs[str(path.relative_to(ROOT))] = sha(path)
    source = (NEW / 'main.tex').read_text()
    inputs[str((NEW / 'main.tex').relative_to(ROOT))] = sha(NEW / 'main.tex')
    source = source.replace(r'\usepackage{amsmath,graphicx,booktabs,array}',r'\usepackage{amsmath,graphicx,booktabs,array,colortbl}')
    source = source.replace(r'\graphicspath{{figures/}}',r'\graphicspath{{figures/current/}{figures/}}'+'\n'+r'\definecolor{ourstint}{RGB}{235,246,242}')
    source = source.replace(r'\input{tables/',r'\input{tables/current/')
    source = source.replace('The revision module operates on a frozen object-mapping backbone.',
        'An earlier fixed-identity study in this project tested whether visible mask evidence could remove contaminated geometry while preserving all object IDs. That operation improved existing shapes but could neither append missing surfaces nor revise identity assignments. The present method addresses those additional decisions; we retain the earlier study as V1 and reserve \\ours{} for the current method.\n\nThe revision module operates on a frozen object-mapping backbone.')
    source = source.replace(r'\section{Evidence-Authorized Object Revision}',
        r'\input{sections/merged/version_scope}'+'\n\n'+r'\section{Evidence-Authorized Object Revision}')
    source = source.replace(r'\subsection{Robot integration}',
        r'\input{sections/merged/indoor_context}'+'\n\n'+r'\subsection{Robot integration}')
    source = source.replace('This is an offline integration comparison.',
        'The original system record also includes six garage/outdoor recordings and image-linked landmark and ground examples. Appendix~\\ref{app:routes} retains their inventory and coordinate-version distinctions. These archived outputs are separate from the current four-method garage comparison.\n\nThis is an offline integration comparison.')
    source = source.replace('The evidence remains limited to two new windows from one dataset,',
        'Evidence for the current revision method remains limited to two new windows from one dataset,')
    source = source.replace('All 24 new-window score records and 628 object-by-method IoU entries were replayed successfully.',
        'All 24 new-window score records and 628 object-by-method IoU entries were replayed successfully. Indoor V1 results retain their own frozen evaluation and provenance; they are not new current-method runs.')
    source = source.replace(r'\begin{thebibliography}{9}',r'\begin{thebibliography}{99}')
    source = source.replace(r'\end{thebibliography}',r'''\bibitem{scenenn} B.-S. Hua et al. SceneNN: A Scene Meshes Dataset with aNNotations. 3DV, 2016. \url{https://hkust-vgd.github.io/scenenn/}.
\bibitem{sam2} N. Ravi et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024. \url{https://arxiv.org/abs/2408.00714}.
\bibitem{gdino} S. Liu et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv:2303.05499, 2023. \url{https://arxiv.org/abs/2303.05499}.
\end{thebibliography}''')
    source = source.replace(r'\end{document}',r'\input{sections/merged/v1_supplement}'+'\n'+r'\input{sections/merged/recording_atlas}'+'\n'+r'\end{document}')
    # Modularize the new material within the established source tree.
    pre, body = source.split(r'\begin{document}', 1)
    main, supplement = body.split(r'\clearpage\nobalance\onecolumn', 1)
    dest = OLD / 'sections/merged'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'current_main.tex').write_text(main+'\n')
    (dest / 'current_supplement.tex').write_text(r'\clearpage\nobalance\onecolumn'+'\n'+supplement.replace(r'\end{document}',''))
    (OLD / 'main.tex').write_text(pre+r'\begin{document}'+'\n'+r'\input{sections/merged/current_main}'+'\n'+r'\input{sections/merged/current_supplement}'+'\n'+r'\end{document}'+'\n')
    (OLD / 'merged_version.json').write_text(json.dumps(dict(date='2026-11-16',primary='full_authorization',
        earlier_variant='visibility1 (V1)',archive=str(ARCHIVE.relative_to(ROOT)),inputs=inputs,
        no_new_experiments=True,scope='One narrative and entry point; separate versions, datasets, metrics and robot map revisions.'),indent=2)+'\n')
    print('Merged current method into original manuscript entry; original source archive verified.')


if __name__ == '__main__':main()
