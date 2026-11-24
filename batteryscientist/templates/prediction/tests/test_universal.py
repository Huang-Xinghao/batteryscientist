"""Cross-domain regression, missing labels, leakage and saved-run reproducibility."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from configuration import parse_args, setting_name
from data_provider.electrolyte import electrolyte_loaders, formulation_key, features, validate_partition
from exp.experiment import Experiment


@pytest.fixture
def electrolyte(tmp_path):
    records = []
    for group in range(9):
        for measurement in range(2):
            records.append({
                'solvents': [{'smiles': 'CCO', 'molar_ratio': 0.7},
                             {'smiles': 'CO', 'molar_ratio': 0.3}],
                'salts': [{'smiles': '[Li+].[Cl-]', 'molar_ratio': 1.0}],
                'salt_molar_ratio': 0.01 * (group+1), 'temperature': 25.0 * measurement,
                'conductivity': group+measurement+1.0, 'conductivity_mask': True,
                'anion_ratio': (group+1)/10 if measurement else 1e9,
                'anion_ratio_mask': bool(measurement),
            })
    path = tmp_path / 'electrolyte.json'
    path.write_text(json.dumps(records))
    split_file = tmp_path / 'splits.json'
    split_file.write_text(json.dumps({'train': list(range(10)), 'val': list(range(10, 14)),
                                      'test': list(range(14, 18))}))
    args = parse_args(['--domain', 'electrolyte', '--root_path', str(path),
                       '--split_file', str(split_file), '--tasks', 'conductivity', 'anion_ratio',
                       '--hidden_dim', '8', '--fingerprint_dim', '8', '--batch_size', '3',
                       '--train_epochs', '1', '--patience', '1', '--no-use_gpu',
                       '--checkpoints', str(tmp_path / 'checkpoints')])
    return path, records, args


def test_formulation_group_ignores_order_and_temperature(electrolyte):
    _, records, _ = electrolyte
    a = records[0]
    b = dict(records[1], solvents=list(reversed(a['solvents'])))
    assert formulation_key(a) == formulation_key(b)
    # Temperature remains a predictive input even though it is excluded from grouping.
    assert not np.array_equal(features(a, 8), features(b, 8))
    b['salt_molar_ratio'] = 0.5
    assert formulation_key(a) != formulation_key(b)


def test_split_rejects_same_formulation_in_different_partitions(electrolyte):
    _, records, _ = electrolyte
    groups = [formulation_key(row) for row in records]
    with pytest.raises(ValueError, match='Formulation leakage'):
        validate_partition({'train': list(range(9)), 'val': list(range(9, 14)),
                            'test': list(range(14, 18))}, groups)


def test_train_only_scaling_and_masks(electrolyte):
    path, records, args = electrolyte
    loaders, state = electrolyte_loaders(args)
    assert state['target_scalers']['conductivity']['mean'] == 3.5
    assert state['target_scalers']['anion_ratio']['mean'] == pytest.approx(0.3)
    sample = loaders[0].dataset[0]
    assert not sample['target_masks']['anion_ratio']
    assert sample['targets']['anion_ratio'] == 0
    # Changing held-out measurements must not affect training scalers or inputs.
    for row in records[10:]:
        row['conductivity'] = 1e8
        row['temperature'] = 1000.0
    path.write_text(json.dumps(records))
    changed, new_state = electrolyte_loaders(args)
    assert state['feature_scaler'] == new_state['feature_scaler']
    assert state['target_scalers'] == new_state['target_scalers']
    torch.testing.assert_close(sample['features'], changed[0].dataset[0]['features'])


def test_masked_labels_do_not_contribute_to_loss(electrolyte):
    _, _, args = electrolyte
    exp = Experiment(args)
    loaders = exp._get_data()
    batch = next(iter(loaders[0]))
    before = exp._loss(batch)
    mask = batch['target_masks']['anion_ratio']
    batch['targets']['anion_ratio'][~mask] = float('nan')
    torch.testing.assert_close(exp._loss(batch), before)


def test_epoch_counts_observed_targets_across_batches(electrolyte):
    _, _, args = electrolyte
    exp = Experiment(args)
    loaders = exp._get_data()

    class ZeroModel(nn.Module):
        def forward(self, x):
            return {task: torch.zeros(len(x), device=x.device) for task in args.tasks}

    exp.model = ZeroModel()
    # Each training target was standardized separately to zero mean/unit variance.
    # A zero predictor has MSE=1 for each task, despite unequal observed counts.
    assert exp._epoch(loaders[0]) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize('tasks', [['conductivity'], ['anion_ratio'], ['conductivity', 'anion_ratio']])
def test_electrolyte_checkpoint_roundtrip_and_masked_metrics(electrolyte, tasks):
    _, _, args = electrolyte
    args.tasks = tasks
    torch.set_num_threads(1)
    exp = Experiment(args)
    exp.train('roundtrip')
    metrics = exp.test('roundtrip')
    path = exp._directory('roundtrip')
    original = {task: np.load(path / f'pred_{task}.npy') for task in tasks}
    # --checkpoint alone restores the complete saved domain/model/data configuration.
    restored_args = parse_args(['--checkpoint', str(path), '--no-use_gpu'])
    restored = Experiment(restored_args)
    assert restored.test('unused', checkpoint=path) == metrics
    for task in tasks:
        np.testing.assert_array_equal(original[task], np.load(path / f'pred_{task}.npy'))
    if 'anion_ratio' in tasks:
        masks = np.load(path / 'mask_anion_ratio.npy')
        true = np.load(path / 'true_anion_ratio.npy')
        assert np.isnan(true[~masks]).all()
        assert metrics['anion_ratio']['observed_values'] == 2


def test_changed_source_refuses_checkpoint(electrolyte):
    path, records, args = electrolyte
    exp = Experiment(args)
    exp.train('source-check')
    records[0]['conductivity'] += 1
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='source data changed'):
        Experiment(args).test('source-check', test=1)


def test_missing_task_labels_fail_explicitly(electrolyte):
    path, records, args = electrolyte
    for row in records:
        row['anion_ratio_mask'] = False
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match='No valid anion_ratio labels in train'):
        electrolyte_loaders(args)


def test_group_split_is_independent_of_training_seed(electrolyte):
    _, _, args = electrolyte
    args.split_file = None
    _, first = electrolyte_loaders(args)
    args.seed = 42
    _, second = electrolyte_loaders(args)
    assert first['split_indices'] == second['split_indices']


def test_config_cli_override_and_incompatible_model(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'domain': 'electrolyte', 'tasks': ['conductivity'],
                                  'hidden_dim': 16, 'batch_size': 8}))
    args = parse_args(['--config', str(config), '--batch_size', '4'])
    assert args.model == 'CompositionMLP' and args.batch_size == 4 and args.hidden_dim == 16
    assert setting_name(args) == setting_name(args)
    with pytest.raises(SystemExit):
        parse_args(['--domain', 'electrolyte', '--model', 'IC2ML'])
    config.write_text(json.dumps({'hidden_dim': 4.5}))
    with pytest.raises(SystemExit):
        parse_args(['--config', str(config)])


def test_staged_cli_output_directory_and_joinable_csv(electrolyte, tmp_path, monkeypatch):
    import csv
    import run

    _, _, args = electrolyte
    torch.set_num_threads(1)
    args.output_dir = str(tmp_path / 'results' / 'seed_2021')
    args.stage = 'train'
    with monkeypatch.context() as context:
        context.setattr(run, 'parse_args', lambda argv=None: args)
        def unexpected_test(*_args, **_kwargs):
            raise AssertionError('Candidate training must only evaluate validation data')
        context.setattr(Experiment, 'test', unexpected_test)
        run.main([])
    path = Path(args.output_dir)
    assert (path / 'checkpoint.pth').is_file()
    validation = json.loads((path / 'validation_metrics.json').read_text())
    assert validation['conductivity']['observed_values'] == 4
    assert not (path / 'metrics.json').exists()
    assert not (path / 'targets.csv').exists()
    validation_before = (path / 'validation_metrics.json').read_bytes()
    checkpoint_before = (path / 'checkpoint.pth').read_bytes()
    run.main(['--stage', 'test', '--checkpoint', str(path), '--no-use_gpu'])
    assert (path / 'validation_metrics.json').read_bytes() == validation_before
    assert (path / 'checkpoint.pth').read_bytes() == checkpoint_before
    metrics = json.loads((path / 'metrics.json').read_text())
    with (path / 'targets.csv').open() as stream:
        targets = list(csv.DictReader(stream))
    with (path / 'predictions.csv').open() as stream:
        predictions = list(csv.DictReader(stream))
    keys = lambda rows: [(r['sample_id'], r['task'], r['target_index']) for r in rows]
    assert keys(targets) == keys(predictions) and len(set(keys(targets))) == len(targets)
    for task in args.tasks:
        rows = [r for r in targets if r['task'] == task and r['observed'] == 'True']
        values = [r for r in predictions if r['task'] == task and r['observed'] == 'True']
        mae = np.mean([abs(float(p['prediction']) - float(t['target'])) for p, t in zip(values, rows)])
        assert mae == pytest.approx(metrics[task]['mae'])
    assert any(r['target'] == '' for r in targets if r['task'] == 'anion_ratio')


def test_evaluation_stage_does_not_change_run_identity(tmp_path):
    train = parse_args(['--stage', 'train', '--output_dir', str(tmp_path / 'run')])
    test = parse_args(['--stage', 'test', '--output_dir', str(tmp_path / 'run')])
    assert setting_name(train) == setting_name(test)


def test_preflight_checks_real_chain_without_fitting_or_test_scoring(electrolyte, tmp_path, monkeypatch):
    _, _, args = electrolyte
    args.output_dir = str(tmp_path / 'preflight')
    torch.set_num_threads(1)
    exp = Experiment(args)
    before = {key: value.clone() for key, value in exp.model.state_dict().items()}

    def unexpected(*_args, **_kwargs):
        raise AssertionError('Preflight must not train or score held-out data')

    monkeypatch.setattr(Experiment, 'train', unexpected)
    monkeypatch.setattr(Experiment, 'test', unexpected)
    result = exp.check('check')
    assert result['passed'] and result['gradient_check'] == 'passed'
    assert not result['test_scored']
    assert result['samples'] == dict(train=10, val=4, test=4)
    assert {p.name for p in Path(args.output_dir).iterdir()} == {'preflight.json'}
    for key, value in exp.model.state_dict().items():
        torch.testing.assert_close(before[key], value, rtol=0, atol=0)


def test_preflight_supports_parameterless_predictors(electrolyte, monkeypatch):
    _, _, args = electrolyte

    class ConstantModel(nn.Module):
        def forward(self, x):
            return {task: torch.zeros(len(x), device=x.device) for task in args.tasks}

    monkeypatch.setattr('exp.experiment.build_model', lambda args: ConstantModel())
    result = Experiment(args).check('constant')
    assert result['passed'] and result['gradient_check'] == 'not_applicable'
    assert 'finite_gradients' not in result['checks']


def test_test_stage_writes_separate_outputs(electrolyte, tmp_path):
    import run

    _, _, args = electrolyte
    train_dir, test_dir = tmp_path / 'train', tmp_path / 'test'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({key: value for key, value in vars(args).items()
                                  if key not in ('config', 'checkpoint')}))
    run.main(['--config', str(config), '--stage', 'train', '--output_dir', str(train_dir)])
    original = {p.name: p.read_bytes() for p in train_dir.iterdir()}
    run.main(['--config', str(config), '--stage', 'test', '--checkpoint', str(train_dir),
              '--output_dir', str(test_dir)])
    assert (test_dir / 'metrics.json').exists()
    assert (test_dir / 'predictions.csv').exists()
    assert not (test_dir / 'checkpoint.pth').exists()
    assert {p.name: p.read_bytes() for p in train_dir.iterdir()} == original


def test_life_only_adapter_accepts_batch_without_capacity_targets():
    from adapters.cell import CellAdapter
    from argparse import Namespace

    adapter = CellAdapter(Namespace(tasks=['life']))

    def model(curves):
        return None, None, torch.ones(len(curves))

    outputs = adapter.forward(model, {'cycle_curves': torch.zeros(2, 2, 3, 20),
                                    'life_label': torch.ones(2)}, torch.device('cpu'))
    assert list(outputs) == ['life']
    assert outputs['life'][2].all()
