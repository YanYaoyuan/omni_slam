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
citations = set()
def inspect_document(name):
    path = root / name
    if not path.suffix:
        path = path.with_suffix('.tex')
    if path in visited:
        return
    visited.add(path)
    source = path.read_text()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', source):
        citations.update(key.strip() for key in group.split(','))
    figures.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source))
    for child in re.findall(r'\\(?:input|include)\{([^}]+)\}', source):
        inspect_document(child)

inspect_document(manifest.get('entry', 'main.tex'))
missing = set(manifest.get('required_figure_references', [])) - figures
if missing:
    raise SystemExit('Required figures absent from active manuscript: ' + ', '.join(sorted(missing)))
if manifest.get('enforce_exact_figure_references'):
    unexpected = figures - set(manifest['required_figure_references'])
    if unexpected:
        raise SystemExit('Figures outside current manuscript scope: ' + ', '.join(sorted(unexpected)))
if manifest.get('bibliography'):
    entries = re.findall(r'@\w+\{([^,]+),', (root / manifest['bibliography']).read_text())
    if len(entries) != len(set(entries)):
        raise SystemExit('Duplicate bibliography keys')
    if citations != set(entries):
        raise SystemExit('Citation mismatch: missing=' + str(sorted(citations - set(entries))) +
                         ', uncited=' + str(sorted(set(entries) - citations)))
print(f"Verified {len(manifest['files_sha256'])} source/assets and {len(figures)} active figure references; this is an integrity check, not a new experiment.")
