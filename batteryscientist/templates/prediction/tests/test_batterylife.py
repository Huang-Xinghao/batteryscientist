"""Protocol checks use tiny PKLs; prediction-quality results require real data."""

from argparse import Namespace
import json
import pickle

import numpy as np
import pytest
import torch

from data_provider.batterylife import BatteryLifeDataset, fixed_splits, cell_location, batterylife_trainloader
from exp.exp_forecasting import Exp_Long_Term_Forecast1


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    (root / 'Tongji').mkdir(parents=True)
    (root / 'Life labels').mkdir()
    names = ['Tongji1_fixture--1.pkl', 'Tongji1_fixture--2.pkl', 'Tongji1_fixture--3.pkl']
    labels = {}
    for name, life in zip(names, [200, 400, 1000]):
        cycles = []
        for i in range(8):
            cycles.append({'cycle_number': 10*(i+1),
                'voltage_in_V': [3.0, 3.4, 3.8, 4.2, 4.1, 3.8, 3.4, 3.0, 2.8, 2.8],
                'current_in_A': [1., 1., 1., 1., 0., -1., -1., -1., -1., 0.],
                'charge_capacity_in_Ah': [0., .5, 1., 2., 0., 0., 0., 0., 0., 0.],
                'discharge_capacity_in_Ah': [0., 0., 0., 0., 0., .5, 1., 1.5, 2.-i*.02, 0.],
                'time_in_s': list(range(10))})
        # Capacities never reach 80%: only the published label may define life.
        data = {'cell_id': name, 'nominal_capacity_in_Ah': 2., 'cycle_data': cycles}
        (root / 'Tongji' / name).write_bytes(pickle.dumps(data))
        labels[name.replace('--', '-#')] = life
    (root / 'Life labels/Tongji_labels.json').write_text(json.dumps(labels))
    args = Namespace(root_path=str(root), dataset='Tongji', condition='all',
        tasks=['soh', 'trajectory', 'rul'], context=2, horizon=2,
        early_cycle_threshold=4, charge_discharge_length=20, hidden_dim=8,
        use_gpu=False, use_multi_gpu=False, gpu=0, model='IC2ML',
        checkpoints=str(tmp_path/'checkpoints'), batch_size=2, num_workers=0,
        train_epochs=1, patience=1, learning_rate=.001, seed=2021)
    return root, names, args


def test_fixed_cell_membership_and_chemistry_subsets():
    splits, _ = fixed_splits('Tongji')
    assert [len(splits[k]) for k in ['train', 'val', 'test']] == [66, 21, 21]
    assert len(set(sum(splits.values(), []))) == 108
    for name, prefix in [('NCA', 'Tongji1_'), ('NCM', 'Tongji2_'), ('NCMNCA', 'Tongji3_')]:
        subset, _ = fixed_splits(name)
        assert subset == {k: [v for v in values if v.startswith(prefix)] for k, values in splits.items()}


def test_published_labels_train_only_scaler_and_record_coordinates(dataset):
    _, names, args = dataset
    train = BatteryLifeDataset(args, 'train', names[:2])
    val = BatteryLifeDataset(args, 'val', names[2:], train.label_scaler)
    assert train.label_scaler is val.label_scaler
    np.testing.assert_allclose(train.label_scaler.mean_, [300.])
    sample = val[0]
    assert sample['life_label'].item() == 7.
    assert sample['life'].item() == 1000.
    assert sample['current_cycle'].item() == 2
    assert sample['raw_cycle_number'].item() == 20
    assert sample['cycle_curves'].shape == (2, 3, 20)
    np.testing.assert_allclose(sample['history_soh'].numpy(), [1., .99])
    np.testing.assert_allclose(sample['future_soh'].numpy(), [.98, .97])
    assert len(val) == 3


def test_missing_labels_are_not_synthesized(dataset):
    root, names, args = dataset
    labels_path = root / 'Life labels/Tongji_labels.json'
    labels = json.loads(labels_path.read_text())
    del labels[names[0].replace('--', '-#')]
    labels_path.write_text(json.dumps(labels))
    train = BatteryLifeDataset(args, 'train', names[:2])
    assert train.cell_records[0]['reason'] == 'missing_published_life_label'
    assert len(train.cells) == 1


def test_soh_only_does_not_require_future_window(dataset):
    _, names, args = dataset
    args.horizon = 500
    args.tasks = ['soh']
    data = BatteryLifeDataset(args, 'train', names[:1])
    assert len(data) == 3 and data[0]['future_soh'].numel() == 0


def test_future_capacity_does_not_leak_into_history(dataset):
    root, names, args = dataset
    before = BatteryLifeDataset(args, 'train', names[:1])[0]
    path = root / 'Tongji' / names[0]
    raw = pickle.loads(path.read_bytes())
    raw['cycle_data'][2]['discharge_capacity_in_Ah'][8] = 1.0
    path.write_bytes(pickle.dumps(raw))
    after = BatteryLifeDataset(args, 'train', names[:1])[0]
    torch.testing.assert_close(before['cycle_curves'], after['cycle_curves'])
    torch.testing.assert_close(before['history_soh'], after['history_soh'])
    assert not torch.equal(before['future_soh'], after['future_soh'])


@pytest.mark.parametrize('tasks', [['soh'], ['trajectory'], ['rul'], ['life'], ['soh', 'trajectory', 'rul']])
def test_only_selected_task_heads_receive_supervision(dataset, tasks):
    _, names, args = dataset
    args.tasks = tasks
    torch.set_num_threads(1)
    data = BatteryLifeDataset(args, 'train', names[:2])
    loader = torch.utils.data.DataLoader(data, batch_size=2)
    exp = Exp_Long_Term_Forecast1(args)
    loss = exp._loss(next(iter(loader)), torch.nn.MSELoss())
    loss.backward()
    for task, head in [('soh', exp.model.soh_predictor),
                       ('trajectory', exp.model.trajectory_predictor),
                       ('rul', exp.model.RUL_predictor)]:
        expected = task in tasks or (task == 'rul' and 'life' in tasks)
        assert any(p.grad is not None for p in head.parameters()) == expected


def test_mich_unmerged_data_matches_batterylife_copy_precedence(tmp_path):
    name = 'MICH_fixture.pkl'
    for folder in ('MICH', 'MICH_EXP'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / name).touch()
    path, labels, key = cell_location(tmp_path, name)
    assert path == tmp_path / 'MICH_EXP' / name
    assert labels == tmp_path / 'Life labels/total_MICH_labels.json'
    assert key == name
    (tmp_path / 'total_MICH').mkdir()
    (tmp_path / 'total_MICH' / name).touch()
    assert cell_location(tmp_path, name)[0] == tmp_path / 'total_MICH' / name


def test_empty_chemistry_split_fails_before_reading_data(dataset):
    _, _, args = dataset
    args.dataset = 'NCMNCA'
    with pytest.raises(ValueError, match='has no cells.*val'):
        batterylife_trainloader(args)


@pytest.mark.parametrize('tasks', [['soh'], ['trajectory'], ['rul'], ['life'], ['soh', 'trajectory', 'rul']])
def test_training_checkpoint_reload_preserves_predictions(dataset, monkeypatch, tasks):
    _, names, args = dataset
    args.tasks = tasks
    torch.set_num_threads(1)
    splits = dict(zip(('train', 'val', 'test'), ([name] for name in names)))
    monkeypatch.setattr('data_provider.batterylife.fixed_splits',
                        lambda *a, **k: (splits, {'source': 'test_fixture'}))
    experiment = Exp_Long_Term_Forecast1(args)
    setting = '-'.join(tasks)
    experiment.train(setting)
    metrics = experiment.test(setting)
    path = experiment._directory(setting)
    predictions = {task: np.load(path / f'pred_{task}.npy') for task in tasks}
    reloaded = Exp_Long_Term_Forecast1(args)
    assert reloaded.test(setting, test=1) == metrics
    for task in tasks:
        np.testing.assert_array_equal(np.load(path / f'pred_{task}.npy'), predictions[task])
    assert set(metrics) == set(tasks)


def test_explicit_split_file_and_cell_sampler_are_used(dataset, tmp_path):
    _, names, args = dataset
    split_file = tmp_path / 'custom.json'
    split_file.write_text(json.dumps(dict(zip(('train', 'val', 'test'), [[name] for name in names]))))
    args.split_file = str(split_file)
    args.sampling = 'cell_balanced'
    train, val, test = batterylife_trainloader(args)
    assert [loader.dataset.files for loader in (train, val, test)] == [[name] for name in names]
    assert train.dataset.split_provenance['source'] == str(split_file)
    assert train.sampler.weights.sum().item() == pytest.approx(1.)
