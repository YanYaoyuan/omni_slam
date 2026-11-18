"""Read-only integrity check of the delivered LaTeX source archive."""
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'source_manifest.json').read_text())
for name, expected in manifest['files_sha256'].items():
    path = root / name
    if not path.is_file():
        raise SystemExit('Missing source: ' + name)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Source changed since delivery: ' + name)
# Follow the active entry point: files present on disk are not necessarily in the PDF.
visited = set()
figures = set()
def inspect_document(name):
    path = root / name
    if not path.suffix:
        path = path.with_suffix('.tex')
    if path in visited:
        return
    visited.add(path)
    source = path.read_text()
    figures.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source))
    for child in re.findall(r'\\(?:input|include)\{([^}]+)\}', source):
        inspect_document(child)

inspect_document(manifest.get('entry', 'main.tex'))
missing = set(manifest.get('required_figure_references', [])) - figures
if missing:
    raise SystemExit('Required figures absent from active manuscript: ' + ', '.join(sorted(missing)))
print(f"Verified {len(manifest['files_sha256'])} source/assets and {len(figures)} active figure references; this is an integrity check, not a new experiment.")
