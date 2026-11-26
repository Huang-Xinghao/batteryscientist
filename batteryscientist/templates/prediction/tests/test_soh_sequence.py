"""A second input protocol uses the production training and checkpoint path."""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from configuration import parse_args
from data_provider.soh_sequence import sequence_loaders
from evaluation import regression_metrics
from exp.experiment import Experiment


@pytest.fixture
def sequence(tmp_path):
    root = tmp_path/'data'
    (root/'fixture').mkdir(parents=True)
    names = []
    for i, length in enumerate([8, 11, 9, 10]):
        name = f'cell_{i}.pkl'; names.append(name)
        # No life labels or current/voltage curves.
        data = dict(cell_id=f'physical_{i}', nominal_capacity_in_Ah=2., cycle_data=[
            dict(cycle_number=j+1, discharge_capacity_in_Ah=[0, 2.-.02*j]) for j in range(length)])
        (root/'fixture'/name).write_bytes(pickle.dumps(data))
    split = tmp_path/'split.json'
    split.write_text(json.dumps(dict(datasets={'fixture': dict(train=names[:2], val=names[2:3], test=names[3:])})))
    args = parse_args(['--adapter', 'soh_sequence', '--dataset', 'fixture', '--root_path', str(root),
                       '--split_file', str(split), '--context', '3', '--horizon', '2',
                       '--aggregation', 'cell_macro', '--selection_metric', 'trajectory.rmse',
                       '--sampling', 'cell_balanced', '--batch_size', '4', '--min_epochs', '1', '--max_epochs', '2',
                       '--patience', '2', '--hidden_dim', '3', '--charge_discharge_length', '3',
                       '--early_cycle_threshold', '1', '--no-use_gpu', '--output_dir', str(tmp_path/'run')])
    return args, root, split, names


def test_sequence_check_train_restore_and_csv(sequence):
    args, _, _, _ = sequence
    torch.set_num_threads(1)
    exp = Experiment(args)
    assert args.model == 'SOHLinear'
    assert exp.check('check')['passed']
    exp.train('train')
    before = exp.test('test')
    path = Path(args.output_dir)
    predictions = np.load(path/'pred_trajectory.npy')
    restored = Experiment(parse_args(['--checkpoint', str(path), '--no-use_gpu']))
    assert restored.test('unused', checkpoint=path) == before
    np.testing.assert_array_equal(predictions, np.load(path/'pred_trajectory.npy'))
    assert before['trajectory']['cells'] == 1
    assert (path/'predictions.csv').is_file()
    assert all(row['selection_metric'] == 'trajectory.rmse'
               for row in json.loads((path/'training_history.json').read_text()))


def test_macro_aggregation_averages_cell_rmse_not_pooled_errors():
    p = np.array([[1.], [1.], [1.], [5.]])
    samples = [dict(cell_id=name) for name in ['a', 'a', 'a', 'b']]
    mask = np.ones_like(p, dtype=bool)
    macro = regression_metrics(p, np.zeros_like(p), mask, samples, 'cell_macro')
    micro = regression_metrics(p, np.zeros_like(p), mask, samples)
    assert macro['rmse'] == 3.
    assert micro['rmse'] == pytest.approx(np.sqrt(7))


def test_membership_sampling_weights_and_no_future_input(sequence):
    args, root, split, names = sequence
    loaders, state = sequence_loaders(args)
    assert [len(loader.dataset) for loader in loaders] == [11, 5, 6]
    weights = loaders[0].sampler.weights.numpy()
    assert weights[:4].sum() == pytest.approx(1.)
    assert weights[4:].sum() == pytest.approx(1.)
    before = loaders[0].dataset[0]
    path = root/'fixture'/names[0]
    data = pickle.loads(path.read_bytes())
    data['cycle_data'][3]['discharge_capacity_in_Ah'] = [0, .5]
    path.write_bytes(pickle.dumps(data))
    after, _ = sequence_loaders(args)
    torch.testing.assert_close(before['history'], after[0].dataset[0]['history'])
    assert not torch.equal(before['target'], after[0].dataset[0]['target'])
    with pytest.raises(ValueError, match='changed since checkpoint'):
        sequence_loaders(args, state)
    split.write_text(json.dumps(dict(train=names[:2], val=[names[0]], test=names[3:])))
    with pytest.raises(ValueError, match='leakage'):
        sequence_loaders(args)


def test_raw_cycle_gaps_and_cell_identity(sequence):
    args, root, _, names = sequence
    args.cycle_coordinate = 'cycle_number'
    path = root/'fixture'/names[0]
    raw = pickle.loads(path.read_bytes())
    raw['cycle_data'][4]['cycle_number'] = 100
    path.write_bytes(pickle.dumps(raw))
    loaders, _ = sequence_loaders(args)
    assert len(loaders[0].dataset) == 7
    raw = pickle.loads((root/'fixture'/names[2]).read_bytes())
    raw['cell_id'] = 'physical_0'
    (root/'fixture'/names[2]).write_bytes(pickle.dumps(raw))
    with pytest.raises(ValueError, match='identity leakage'):
        sequence_loaders(args)


def test_checkpoint_selects_configured_metric_instead_of_loss(sequence, monkeypatch):
    args, _, _, _ = sequence
    exp = Experiment(args)
    epoch = [0]
    def fake_epoch(loader, optimizer=None):
        if optimizer is not None:
            epoch[0] += 1
            with torch.no_grad():
                for parameter in exp.model.parameters():
                    parameter.fill_(epoch[0])
        return epoch[0]
    monkeypatch.setattr(exp, '_epoch', fake_epoch)
    monkeypatch.setattr(exp, '_evaluate', lambda *a, **kw: {'trajectory': {'rmse': 3.-epoch[0]}})
    exp.train('selection')
    for parameter in exp.model.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 2.))


@pytest.mark.parametrize('model', ['SOHPersistence', 'SOHTrend', 'SOHMLP', 'SOHTCN'])
def test_baselines_share_pipeline_and_checkpoint(sequence, model):
    args, _, _, _ = sequence
    args.model = model
    args.protocol_file = str(Path(args.output_dir).parent/'comparison.json')
    exp = Experiment(args)
    assert exp.check('check')['passed']
    exp.train('train')
    metrics = exp.test('test')
    assert np.isfinite(metrics['trajectory']['rmse'])
    assert metrics['trajectory']['windows'] == 6
    if model in ('SOHPersistence', 'SOHTrend'):
        assert json.loads((Path(args.output_dir)/'fit_status.json').read_text())['fitted'] is False
    restored = Experiment(parse_args(['--checkpoint', args.output_dir, '--no-use_gpu']))
    assert restored.test('test', checkpoint=args.output_dir) == metrics


def test_deterministic_baseline_predictions_use_only_observed_history(sequence):
    from models.SOHBaselines import Persistence, Trend
    args, _, _, _ = sequence
    history = torch.tensor([[1., .9, .8], [2., 2., 2.]])
    torch.testing.assert_close(Persistence(args)(history), torch.tensor([[.8, .8], [2., 2.]]))
    torch.testing.assert_close(Trend(args)(history), torch.tensor([[.7, .6], [2., 2.]]))


def test_comparison_protocol_allows_models_and_seeds_but_rejects_changed_population(sequence):
    from copy import deepcopy
    args, _, _, _ = sequence
    args.protocol_file = str(Path(args.output_dir).parent/'protocol.json')
    original = Experiment(args)
    original._get_data()
    record = Path(args.protocol_file).read_bytes()
    alternate = deepcopy(args)
    alternate.model = 'SOHTCN'
    alternate.seed = 42
    alternate.hidden_dim = 8
    alternate.learning_rate = .01
    Experiment(alternate)._get_data()
    assert Path(args.protocol_file).read_bytes() == record
    alternate.horizon += 1
    with pytest.raises(ValueError, match='Comparison protocol changed'):
        Experiment(alternate)._get_data()
    alternate.horizon = args.horizon
    alternate.aggregation = 'observation'
    with pytest.raises(ValueError, match='Comparison protocol changed'):
        Experiment(alternate)._get_data()


def test_training_seed_does_not_change_subsampled_cell_membership(sequence):
    args, _, _, _ = sequence
    args.dataaccess = 50
    args.seed = 17
    _, before = sequence_loaders(args)
    args.seed = 43
    _, after = sequence_loaders(args)
    assert before == after


def test_test_never_scores_random_unrestored_weights(sequence):
    args, _, _, _ = sequence
    with pytest.raises(FileNotFoundError):
        Experiment(args).test('untrained')


def test_causal_tcn_block_does_not_read_later_positions():
    from models.SOHBaselines import CausalBlock
    block = CausalBlock(1, 3, 2).eval()
    history = torch.randn(2, 1, 12)
    changed = history.clone()
    changed[:, :, 7:] += 100
    torch.testing.assert_close(block(history)[:, :, :7], block(changed)[:, :, :7], rtol=0, atol=0)


def test_evaluation_rejects_broadcastable_wrong_target_shape(sequence):
    args, _, _, _ = sequence
    exp = Experiment(args)
    class WrongShape(torch.nn.Module):
        def forward(self, history):
            return history[:, -1:]
    exp.model = WrongShape()
    with pytest.raises(ValueError, match='shape mismatch'):
        exp.validate('invalid')
