"""The distributed template, not just its source tree, supports SOH histories."""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from batteryscientist.tools.core import ToolContext
from batteryscientist.tools.experiment_template import prepare_experiment_template, prepare_template_run


@pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='Template execution needs torch')
def test_packaged_soh_template_runs_standard_jobs(tmp_path):
    root = tmp_path/'data'/'example'
    root.mkdir(parents=True)
    names = []
    for cell in range(3):
        name = f'example_{cell}.pkl'; names.append(name)
        (root/name).write_bytes(pickle.dumps(dict(cell_id=name, nominal_capacity_in_Ah=2., cycle_data=[
            dict(cycle_number=i+1, discharge_capacity_in_Ah=[0, 2.-i*.01]) for i in range(7)])))
    (tmp_path/'splits.json').write_text(json.dumps(dict(zip(('train', 'val', 'test'), [[n] for n in names]))))
    ctx = ToolContext(tmp_path)
    copied = prepare_experiment_template(ctx, domain='cell', adapter='soh_sequence', dataset='example',
                                         data_path='data', split_file='splits.json')
    assert not copied.is_error, copied.output
    info = copied.structured_content
    config_path = tmp_path/info['config']
    config = json.loads(config_path.read_text())
    assert config['model'] == 'SOHLinear'
    assert config['selection_metric'] == 'trajectory.rmse'
    config.update(context=2, horizon=2, min_epochs=1, max_epochs=1, patience=1, batch_size=2, use_gpu=False)
    config_path.write_text(json.dumps(config))
    queued = prepare_template_run(ctx, name='history', device='cpu', include_test=True)
    assert not queued.is_error, queued.output
    manifest = json.loads((tmp_path/'experiments/manifest.json').read_text())
    for job in manifest['jobs']:
        argv = [sys.executable, *job['argv'][1:]]
        result = subprocess.run(argv, cwd=tmp_path, env=dict(os.environ, BATTERYSCIENTIST_DEVICE='cpu',
                                                          OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        for output in job['expected_outputs']:
            assert (tmp_path/output['path']).is_file()
    output = tmp_path/queued.structured_content['outputs']
    assert json.loads((output/'check/preflight.json').read_text())['passed']
    metrics = json.loads((output/'test/metrics.json').read_text())
    assert metrics['trajectory']['aggregation'] == 'cell_macro'
    assert not (output/'train/metrics.json').exists()


def test_builtin_cell_membership_is_snapshotted_with_source(tmp_path):
    ctx = ToolContext(tmp_path)
    result = prepare_experiment_template(ctx, domain='cell', adapter='soh_sequence', dataset='HUST')
    assert not result.is_error, result.output
    info = result.structured_content
    assert info['split_counts'] == {'train': 47, 'val': 15, 'test': 15}
    snapshot = json.loads(Path(info['split_file']).read_text())
    assert snapshot['provenance']['source_commit']
    membership = snapshot['datasets']['HUST']
    assert len(set(sum(membership.values(), []))) == 77
    assert info['baseline_models'] == ['SOHPersistence', 'SOHTrend', 'SOHLinear', 'SOHMLP', 'SOHTCN']


def test_custom_split_snapshot_does_not_follow_later_source_edits(tmp_path):
    source = tmp_path/'split.json'
    original = {'train': ['a.pkl'], 'val': ['b.pkl'], 'test': ['c.pkl']}
    source.write_text(json.dumps(original))
    result = prepare_experiment_template(ToolContext(tmp_path), domain='cell', dataset='custom',
                                         split_file='split.json')
    assert not result.is_error, result.output
    source.write_text('{}')
    snapshot = json.loads(Path(result.structured_content['split_file']).read_text())
    assert snapshot['datasets']['custom'] == original


def test_invalid_cell_membership_fails_before_publishing_template(tmp_path):
    source = tmp_path/'split.json'
    source.write_text(json.dumps({'train': ['a.pkl'], 'val': ['a.pkl'], 'test': ['c.pkl']}))
    result = prepare_experiment_template(ToolContext(tmp_path), domain='cell', dataset='custom',
                                         split_file='split.json')
    assert result.is_error and 'leakage' in result.output
    assert not (tmp_path/'experiments/template').exists()
