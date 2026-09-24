"""Characterize the upstream scientific generator and complete cell coverage."""
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['Generated_Flow.ipynb', 'Real_Flow.ipynb'])
def test_every_scientific_cell_is_tracked(name):
    notebook = json.loads((ROOT/name).read_text())
    cells = notebook['cells']
    assert sum('dataerai-setup' in c.get('metadata',{}).get('tags',[]) for c in cells) == 1
    assert sum('dataerai-finish' in c.get('metadata',{}).get('tags',[]) for c in cells) == 1
    for cell in cells:
        if cell['cell_type'] != 'code': continue
        tags = cell.get('metadata',{}).get('tags',[])
        if any(tag in tags for tag in ['dataerai-setup','dataerai-finish']): continue
        assert ''.join(cell['source']).startswith('%%dataerai\n')
        assert not cell['outputs']


def test_generated_data_is_repeatable_and_has_expected_physical_range():
    notebook = json.loads((ROOT/'Generated_Flow.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if any(header in source for header in ['# Imports for Generating & Viewing Data #', '# Constants #', '# Utility Functions for Generating Data #']):
            exec(source.removeprefix('%%dataerai\n'), namespace)
    first = namespace['gen_data'](2)
    namespace['rng'] = np.random.default_rng(namespace['SEED'])
    second = namespace['gen_data'](2)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    images = first[0]
    side = 320
    assert images.shape == (2, side, side, 1)
    assert images.dtype == np.float32
    assert 0 <= images.min() <= images.max() < 1


def test_smoke_parameters_do_not_change_full_defaults(monkeypatch):
    from rheed_runtime import parameter
    monkeypatch.delenv('RHEED_NUM_EPOCHS', raising=False)
    assert parameter('NUM_EPOCHS',100) == 100
    monkeypatch.setenv('RHEED_NUM_EPOCHS','1')
    assert parameter('NUM_EPOCHS',100) == 1
    monkeypatch.setenv('RHEED_NUM_EPOCHS','0')
    with pytest.raises(ValueError): parameter('NUM_EPOCHS',100)
