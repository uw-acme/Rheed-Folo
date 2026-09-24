import json
from pathlib import Path
import subprocess
import sys
import nbformat
import pytest


@pytest.mark.parametrize("capture_failure", [False, True])
def test_runner_persists_error_and_calls_finalizer(tmp_path, capture_failure):
    """A real Jupyter kernel proves failure cleanup, without Dataerai/network access."""
    root = Path(__file__).resolve().parents[1]
    path = tmp_path/'failure.ipynb'
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell('from pathlib import Path\nclass Recorder:\n    run_id="run"\n    run_asset_id="run-asset"\n    summary_asset_id="summary"\n    run_dir=Path.cwd()\n    cell_count=1\n    asset_count=1\n    relationship_count=1\n    def capture_file(self,path,**kwargs):\n        Path("captured.txt").write_text(Path(path).read_text())\n    def finish(self,**kwargs):\n        Path("finalized.txt").write_text(kwargs["status"])\ndataerai=Recorder()'),
        nbformat.v4.new_code_cell('raise ValueError("expected failure")'),
        nbformat.v4.new_code_cell('Path("must-not-run").touch()'),
    ]
    if capture_failure:
        nb.cells[0].source = nb.cells[0].source.replace('Path("captured.txt").write_text(Path(path).read_text())', 'raise RuntimeError("transfer unavailable")')
    nbformat.write(nb,path)
    result = subprocess.run([sys.executable,str(root/'run_dataerai.py'),str(path),'--timeout','60'],capture_output=True,text=True)
    assert result.returncode != 0
    assert (tmp_path/'finalized.txt').read_text() == 'failed'
    assert not (tmp_path/'must-not-run').exists()
    executed_path = next(tmp_path.glob('.dataerai/executions/*/failure.ipynb')) if capture_failure else tmp_path/'captured.txt'
    executed = json.loads(executed_path.read_text())
    assert executed['cells'][1]['outputs'][0]['ename'] == 'ValueError'
    assert executed['metadata']['dataerai_execution']['status'] == 'failed'


def test_archive_retains_paths_bytes_and_checksums(tmp_path):
    import hashlib
    import zipfile
    from rheed_runtime import capture_directory
    folder = tmp_path/'generated'
    (folder/'nested').mkdir(parents=True)
    (folder/'nested/model.bin').write_bytes(b'\x01\x02\x03')
    class Recorder:
        run_dir = tmp_path
        def capture_file(self,path,**kwargs): return path
    path = capture_directory(Recorder(),folder)
    with zipfile.ZipFile(path) as z:
        assert z.read('nested/model.bin') == b'\x01\x02\x03'
        manifest=json.loads(z.read('dataerai-file-manifest.json'))
        assert manifest[0]['sha256'] == hashlib.sha256(b'\x01\x02\x03').hexdigest()


def test_runner_accepts_collection_prefix_and_postfix(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / 'collection-routing.ipynb'
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(
        'import os\nfrom pathlib import Path\n'
        'Path("collection-routing.json").write_text(__import__("json").dumps({'
        '"prefix": os.environ.get("DATAERAI_COLLECTION_PREFIX"), '
        '"postfix": os.environ.get("DATAERAI_COLLECTION_POSTFIX")}))'
    )]
    nbformat.write(nb, path)

    result = subprocess.run([
        sys.executable, str(root / 'run_dataerai.py'), str(path),
        '--collection-prefix', 'September batch',
        '--collection-postfix', 'rerun 2',
        '--timeout', '60',
    ], cwd=tmp_path, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / 'collection-routing.json').read_text()) == {
        'prefix': 'September batch', 'postfix': 'rerun 2'
    }
