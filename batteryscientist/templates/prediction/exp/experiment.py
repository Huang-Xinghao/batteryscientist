"""Shared supervised-regression runner. Domain semantics live in adapters/."""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from adapters import make_adapter
from models.registry import build_model
from protocol import verify_protocol
from utils.tools import EarlyStopping, adjust_learning_rate


class Experiment:
    def __init__(self, args):
        self.args = args
        self.adapter = make_adapter(args)
        self.tasks = self.adapter.tasks
        if args.model not in self.adapter.compatible_models:
            raise ValueError(f'Model {args.model!r} does not support {self.adapter.domain} inputs; '
                             f'choose from {self.adapter.compatible_models}')
        self.weights = dict(self.adapter.weights)
        overrides = getattr(args, 'task_weights', {})
        if set(overrides) - set(self.tasks):
            raise ValueError('task_weights contains an unselected task')
        self.weights.update(overrides)
        if any(not np.isfinite(w) or w <= 0 for w in self.weights.values()):
            raise ValueError('Task weights must be finite and positive')
        self._loaders = None
        if args.use_gpu and torch.cuda.is_available():
            # --gpu indexes devices visible to this process.
            self.device = torch.device(f'cuda:{args.gpu}')
        else:
            self.device = torch.device('cpu')
        allocated = __import__('os').environ.get('BATTERYSCIENTIST_DEVICE')
        if allocated in {'cpu', 'cuda', 'mps'}:
            if allocated == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('Runner allocated CUDA but PyTorch cannot use it')
            self.device = torch.device('cuda:0' if allocated == 'cuda' else allocated)
        print(f'Use {self.device}')
        self.model = build_model(args).to(self.device)
        from os import environ
        if report := environ.get('BATTERYSCIENTIST_BACKEND_REPORT'):
            Path(report).write_text(json.dumps({'device': self.device.type,
                'framework': 'torch', 'evidence': f'Model parameters allocated on {self.device}'}))

    def _get_data(self, state=None):
        if self._loaders is None:
            self._loaders = self.adapter.loaders(state)
            self.protocol = verify_protocol(self.args, self.adapter)
        return self._loaders

    def _directory(self, setting):
        if getattr(self.args, 'output_dir', None):
            return Path(self.args.output_dir)
        return Path(self.args.checkpoints) / self.args.model / self.args.dataset / self.args.condition / setting

    def _parts(self, batch):
        parts = {}
        for task, (prediction, target, mask) in self.adapter.forward(self.model, batch, self.device).items():
            if prediction.shape != target.shape or mask.shape != target.shape:
                raise ValueError(f'Prediction/target/mask shape mismatch for {task}')
            if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
                raise ValueError(f'Non-finite prediction or observed target for {task}')
            count = int(mask.sum())
            if count:
                parts[task] = self.adapter.loss_parts(task, prediction, target, mask)
        return parts

    def _loss(self, batch, criterion=None):
        # criterion is retained for callers of the previous experiment interface.
        parts = self._parts(batch)
        if not parts:
            raise ValueError('Batch has no observed labels for the selected tasks')
        return sum(self.weights[task] * value / count for task, (value, count) in parts.items())

    def _epoch(self, loader, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        totals = {task: [0.0, 0] for task in self.tasks}
        with torch.set_grad_enabled(training):
            for batch in loader:
                if training:
                    optimizer.zero_grad()
                parts = self._parts(batch)
                if not parts:
                    continue
                loss = sum(self.weights[task] * value / count
                           for task, (value, count) in parts.items())
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite task loss')
                if training:
                    loss.backward()
                    optimizer.step()
                for task, (value, count) in parts.items():
                    totals[task][0] += value.detach().item()
                    totals[task][1] += count
        missing = [task for task, (_, count) in totals.items() if not count]
        if missing:
            raise ValueError(f'No observed labels for tasks {missing} in this epoch')
        return sum(self.weights[task] * value / count for task, (value, count) in totals.items())

    def _metadata(self):
        self._get_data()
        return {'schema_version': 2, 'domain': self.adapter.domain, 'model': self.args.model,
                'tasks': list(self.tasks), 'task_weights': self.weights,
                'arguments': vars(self.args), 'data': self.adapter.state(),
                'comparison_protocol': self.protocol}

    def check(self, setting):
        """Check real data, supervised loss, gradients and disk restore; no fitting or test scoring."""
        train, val, test = self._get_data()
        batch = next(iter(train))
        loss = self._loss(batch)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite preflight loss')
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        checks = ['production_loader_adapter_loss']
        if parameters:
            if not loss.requires_grad:
                raise ValueError('Loss is disconnected from trainable model parameters')
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            observed = [g for g in gradients if g is not None]
            if not observed or any(not torch.isfinite(g).all() for g in observed):
                raise ValueError('Model has no supervised gradients or has non-finite gradients')
            checks.append('finite_gradients')
        self.model.eval()
        with torch.no_grad():
            before = {task: parts[0].clone() for task, parts in
                      self.adapter.forward(self.model, batch, self.device).items()}
        state = self._metadata()
        with tempfile.TemporaryDirectory(prefix='template-check-') as temporary:
            path = Path(temporary)
            (path / 'data_manifest.json').write_text(json.dumps(state))
            torch.save(self.model.state_dict(), path / 'checkpoint.pth')
            restored = Experiment(self.args)
            restored._load_checkpoint(path)
            restored.model.eval()
            if restored.adapter.state() != state['data']:
                raise ValueError('Preprocessing changed after restoring saved state')
            with torch.no_grad():
                for task, parts in restored.adapter.forward(restored.model, batch, restored.device).items():
                    torch.testing.assert_close(before[task], parts[0], rtol=0, atol=0)
        result = {'passed': True, 'domain': self.adapter.domain, 'tasks': list(self.tasks),
                  'checks': checks + ['saved_preprocessing_restore', 'disk_checkpoint_prediction_roundtrip'],
                  'gradient_check': 'passed' if parameters else 'not_applicable',
                  'samples': dict(zip(('train', 'val', 'test'),
                                      (len(loader.dataset) for loader in (train, val, test)))),
                  'test_scored': False}
        destination = self._directory(setting)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'preflight.json').write_text(json.dumps(result, indent=2))
        return result

    def train(self, setting):
        train, val, _ = self._get_data()
        path = self._directory(setting)
        path.mkdir(parents=True, exist_ok=True)
        if (path / 'checkpoint.pth').exists():
            raise FileExistsError(f'Checkpoint already exists; use a new run name: {path}')
        (path / 'data_manifest.json').write_text(json.dumps(self._metadata(), indent=2, ensure_ascii=False))
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            # Deterministic baselines still publish a restorable artifact and
            # train/validation receipts, through the same runner jobs.
            torch.save(self.model.state_dict(), path / 'checkpoint.pth')
            (path / 'training_history.json').write_text('[]\n')
            (path / 'fit_status.json').write_text(json.dumps({
                'fitted': False, 'reason': 'deterministic baseline; no trainable parameters'}))
            return self.model
        optimizer = self.adapter.optimizer(self.model)
        stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        history = []
        for epoch in range(self.args.max_epochs):
            training, validation = self._epoch(train, optimizer), self._epoch(val)
            metric_name = getattr(self.args, 'selection_metric', 'loss')
            score = validation
            if metric_name != 'loss':
                task, separator, metric = metric_name.partition('.')
                if not separator:
                    raise ValueError('selection_metric must be loss or task.metric')
                values = self._evaluate(setting, 'val', False, None, write_outputs=False)
                score = values[task][metric]
                if not isinstance(score, (int, float)) or not np.isfinite(score):
                    raise ValueError(f'Invalid checkpoint selection metric {metric_name}')
            history.append({'epoch': epoch+1, 'selection_metric': metric_name, 'selection_value': score,
                            'train_loss': training, 'val_loss': validation,
                            'learning_rate': optimizer.param_groups[0]['lr']})
            print(f'Epoch {epoch+1}: train={training:.7f}, validation={validation:.7f}')
            stopping(score, self.model, str(path))
            (path / 'training_history.json').write_text(json.dumps(history, indent=2, allow_nan=False))
            if epoch + 1 >= self.args.min_epochs and stopping.early_stop:
                break
            if getattr(self.args, 'lradj', 'constant') != 'constant':
                adjust_learning_rate(optimizer, epoch+2, self.args)
        self.model.load_state_dict(torch.load(path / 'checkpoint.pth', map_location=self.device, weights_only=True))
        return self.model

    def vali(self, train_loader, val_loader, criterion=None):
        return self._epoch(val_loader)

    def _load_checkpoint(self, path):
        metadata = json.loads((path / 'data_manifest.json').read_text())
        # Accept checkpoints created before the generic runner was introduced.
        if metadata.get('domain', 'cell') != self.adapter.domain:
            raise ValueError('Checkpoint domain differs from requested domain')
        if metadata.get('model', metadata['arguments']['model']) != self.args.model:
            raise ValueError('Checkpoint model differs from requested model')
        if metadata['tasks'] != list(self.tasks):
            raise ValueError('Checkpoint task selection differs from requested tasks')
        if metadata.get('task_weights', self.adapter.weights) != self.weights:
            raise ValueError('Checkpoint task weights differ from requested weights')
        model_keys = getattr(type(self.model), 'config_keys', ())
        for name in (*self.adapter.config_keys, *model_keys, 'adapter', 'aggregation'):
            if name in metadata['arguments'] and metadata['arguments'][name] != getattr(self.args, name, None):
                raise ValueError(f'Checkpoint data/model configuration mismatch: {name}')
        data = metadata.get('data', metadata)
        self._loaders = None
        self._get_data(data)
        current = self.adapter.state()
        for flag in ('train', 'val', 'test'):
            if data['splits'][flag] != current['splits'][flag]:
                raise ValueError(f'Checkpoint data split or labels changed: {flag}')
        self.model.load_state_dict(torch.load(path / 'checkpoint.pth', map_location=self.device, weights_only=True))

    def test(self, setting, test=0, checkpoint=None):
        return self._evaluate(setting, 'test', True, checkpoint)

    def validate(self, setting, restore=False, checkpoint=None):
        return self._evaluate(setting, 'val', bool(restore or checkpoint), checkpoint)

    def _evaluate(self, setting, split, restore, checkpoint, write_outputs=True):
        source = Path(checkpoint) if checkpoint else self._directory(setting)
        if restore:
            self._load_checkpoint(source)
        path = self._directory(setting) if getattr(self.args, 'output_dir', None) else source
        path.mkdir(parents=True, exist_ok=True)
        loader = self._get_data()[1 if split == 'val' else 2]
        prefix = 'validation_' if split == 'val' else ''
        self.model.eval()
        predictions, truths, masks = ({task: [] for task in self.tasks} for _ in range(3))
        samples, extras = [], {}
        with torch.no_grad():
            for batch in loader:
                decoded = {}
                for task, (prediction, target, mask) in self.adapter.forward(self.model, batch, self.device).items():
                    if prediction.shape != target.shape or mask.shape != target.shape:
                        raise ValueError(f'Prediction/target/mask shape mismatch for {task}')
                    prediction, target = self.adapter.decode(
                        task, prediction.cpu().numpy(), target.cpu().numpy(), batch)
                    observed = mask.cpu().numpy().astype(bool)
                    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
                        raise ValueError(f'Non-finite evaluation values for {task}')
                    predictions[task].append(prediction)
                    truths[task].append(np.where(observed, target, np.nan))
                    masks[task].append(observed)
                    decoded[task] = (prediction, target)
                samples.extend(self.adapter.samples(batch))
                for name, (prediction, target) in self.adapter.extras(decoded, batch).items():
                    extras.setdefault(name, [[], []])
                    extras[name][0].append(prediction)
                    extras[name][1].append(target)
        metrics = {}
        for task in self.tasks:
            prediction, target, observed = (np.concatenate(values[task]) for values in (predictions, truths, masks))
            if not observed.any():
                raise ValueError(f'No observed {split} labels for {task}')
            metrics[task] = dict(self.adapter.metrics(task, prediction, target, observed, samples),
                                 unit=self.adapter.units[task], windows=len(target),
                                 observed_values=int(observed.sum()))
            if not write_outputs:
                continue
            np.save(path / f'{prefix}pred_{task}.npy', prediction)
            np.save(path / f'{prefix}true_{task}.npy', target)
            np.save(path / f'{prefix}mask_{task}.npy', observed)
            self._save_csv(path, prefix, task, samples, prediction, target, observed)
        if not write_outputs:
            return metrics
        for name, (prediction, target) in extras.items():
            np.save(path / f'{prefix}pred_{name}.npy', np.concatenate(prediction))
            np.save(path / f'{prefix}true_{name}.npy', np.concatenate(target))
        (path / ('validation_samples.json' if split == 'val' else 'test_samples.json')).write_text(json.dumps(samples, indent=2))
        (path / f'{prefix}metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False))
        print(json.dumps(metrics, indent=2))
        return metrics

    def _save_csv(self, path, prefix, task, samples, prediction, target, observed):
        # Composite keys retain the sample and target coordinate for vector tasks.
        mode = 'w' if task == self.tasks[0] else 'a'
        for filename, value_name, values in (
            ('targets.csv', 'target', target), ('predictions.csv', 'prediction', prediction),
        ):
            with (path / f'{prefix}{filename}').open(mode, newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                if mode == 'w':
                    writer.writerow(['sample_id', 'task', 'target_index', 'observed', value_name,
                                     'cell_id', 'current_record', 'raw_cycle_number', 'formulation_group'])
                for sample, row, mask in zip(samples, values, observed):
                    for index, (value, present) in enumerate(zip(np.ravel(row), np.ravel(mask))):
                        writer.writerow([sample['sample_id'], task, index, bool(present),
                                         float(value) if value_name == 'prediction' or present else '',
                                         *[sample.get(key, '') for key in ('cell_id', 'current_record',
                                                                          'raw_cycle_number', 'formulation_group')]])
