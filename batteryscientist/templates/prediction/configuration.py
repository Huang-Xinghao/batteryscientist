"""JSON experiment configurations with command-line overrides."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from adapters import ADAPTERS, ADAPTER_REGISTRY, adapter_class
from models.registry import validate_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Cell and electrolyte property prediction experiments')
    parser.add_argument('--config', help='JSON configuration; explicit CLI options take precedence')
    parser.add_argument('--domain', choices=tuple(ADAPTERS), default='cell')
    parser.add_argument('--protocol_file', help='Shared comparison protocol JSON, created on first use')
    parser.add_argument('--adapter', choices=tuple(ADAPTER_REGISTRY))
    parser.add_argument('--aggregation', choices=['observation', 'cell_macro'], default='observation')
    parser.add_argument('--selection_metric', default='loss', help='Validation loss or task.metric, minimized')
    parser.add_argument('--sampling', choices=['window', 'cell_balanced'], default='window')
    parser.add_argument('--cycle_coordinate', choices=['record', 'cycle_number'], default='record')
    parser.add_argument('--cycle_step', type=float, default=1.0)
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--root_path')
    parser.add_argument('--condition', default='all')
    parser.add_argument('--is_training', type=int, choices=[0, 1], default=1)
    parser.add_argument('--stage', choices=['check', 'train', 'validate', 'test', 'train_test'],
                        help='check validates the data/model chain; train saves validation; test restores a checkpoint')
    parser.add_argument('--output_dir', help='Exact directory for one job checkpoint, configuration and results')
    parser.add_argument('--checkpoint', help='Existing run directory; restores its saved configuration for evaluation')
    parser.add_argument('--run_name', help='Optional experiment directory name')
    parser.add_argument('--checkpoints', default='./checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_epochs', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--loss', choices=['MSE', 'MAE', 'Huber'], default='MSE')
    parser.add_argument('--task_weights', type=json.loads, default={})
    parser.add_argument('--lradj', choices=['constant', 'type1', 'type2', 'cosine'], default='constant')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    # BatteryLife / IC2ML options.
    parser.add_argument('--context', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--early_cycle_threshold', type=int, default=100)
    parser.add_argument('--charge_discharge_length', type=int, default=300)
    parser.add_argument('--dataaccess', type=int, default=100)
    parser.add_argument('--eval_cycle_min', type=int)
    parser.add_argument('--eval_cycle_max', type=int)
    # Electrolyte property baseline / grouping options.
    parser.add_argument('--fingerprint_dim', type=int, default=128)
    parser.add_argument('--split_seed', type=int, default=2021)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--test_fraction', type=float, default=0.2)
    parser.add_argument('--split_file', help='JSON with train/val/test cell filenames (cell) or row indices (electrolyte)')
    # Recognize previous CLI flags, with explicit rejection of unsupported execution modes.
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', default='0')
    parser.add_argument('--des', default='test')
    parser.add_argument('--inverse', default='no')

    preliminary, _ = parser.parse_known_args(argv)
    known = {action.dest for action in parser._actions}
    defaults = {}
    if preliminary.checkpoint:
        manifest = Path(preliminary.checkpoint) / 'data_manifest.json'
        saved = json.loads(manifest.read_text())['arguments']
        defaults.update({key: value for key, value in saved.items()
                         if key in known and key not in ('config', 'checkpoint', 'run_name', 'is_training', 'stage', 'itr')})
        defaults.update(is_training=0, itr=1)
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text())
        if not isinstance(config, dict) or set(config) - known:
            parser.error('Config must be a JSON object containing known option names')
        if set(config) & {'config', 'checkpoint'}:
            parser.error('Pass --config and --checkpoint on the command line')
        defaults.update(config)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    # argparse does not apply choices/type validation to non-string JSON defaults.
    for action in parser._actions:
        value = getattr(args, action.dest, None)
        if action.choices and value is not None and value not in action.choices:
            parser.error(f'{action.dest} must be one of {action.choices}')
        if action.type in (int, float) and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                parser.error(f'{action.dest} must be a finite number')
            if action.type is int and not isinstance(value, int):
                parser.error(f'{action.dest} must be an integer')
    for name in ('use_gpu', 'use_amp', 'use_multi_gpu'):
        if not isinstance(getattr(args, name), bool):
            parser.error(f'{name} must be a boolean')
    for name in ('condition', 'checkpoints', 'des', 'inverse'):
        if not isinstance(getattr(args, name), str):
            parser.error(f'{name} must be a string')
    for name in ('root_path', 'dataset', 'model', 'run_name', 'split_file', 'output_dir', 'protocol_file'):
        if getattr(args, name) is not None and not isinstance(getattr(args, name), str):
            parser.error(f'{name} must be a string')
    try:
        adapter = adapter_class(args)
    except ValueError as error:
        parser.error(str(error))
    args.dataset = args.dataset or ('Tongji' if args.domain == 'cell' else 'bamboo_mixer')
    args.model = args.model or adapter.default_model
    if args.tasks is None:
        args.tasks = list(adapter.default_tasks)
    if not isinstance(args.tasks, list) or not args.tasks or any(not isinstance(t, str) for t in args.tasks):
        parser.error('tasks must be a nonempty list of task names')
    args.tasks = list(dict.fromkeys(args.tasks))
    args.root_path = args.root_path or os.environ.get(
        'BATTERYLIFE_ROOT' if args.domain == 'cell' else 'ELECTROLYTE_ROOT', './dataset')
    if args.model not in adapter.compatible_models:
        parser.error(f'{args.domain} supports models {adapter.compatible_models}')
    try:
        adapter(args)
    except ValueError as error:
        parser.error(str(error))
    if not isinstance(args.task_weights, dict) or set(args.task_weights) - set(args.tasks):
        parser.error('task_weights must map selected task names to positive numbers')
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
           for w in args.task_weights.values()):
        parser.error('Task weights must be finite and positive')
    if min(args.hidden_dim, args.batch_size, args.train_epochs, args.patience, args.itr, args.fingerprint_dim) < 1:
        parser.error('Dimensions, batch size, epochs, patience and itr must be positive')
    if args.num_workers < 0 or args.gpu < 0 or args.seed < 0 or args.split_seed < 0 or args.learning_rate <= 0:
        parser.error('Workers, device and seeds must be nonnegative; learning_rate must be positive')
    if args.use_amp or args.use_multi_gpu:
        parser.error('AMP and multi-GPU are not implemented; use a single device')
    args.stage = args.stage or ('train_test' if args.is_training else 'test')
    args.is_training = int(args.stage in ('train', 'train_test'))
    if args.checkpoint and (args.is_training or args.stage == 'check'):
        parser.error('Use --stage validate or --stage test with an existing checkpoint')
    if args.output_dir is not None and not args.output_dir.strip():
        parser.error('output_dir must be a nonempty directory path')
    if args.output_dir and args.itr > 1:
        parser.error('Use one seed per output_dir; set itr=1 and create a job per seed')
    try:
        validate_model(args)
    except ValueError as error:
        parser.error(str(error))
    if args.context < 1 or args.horizon < 1 or args.cycle_step <= 0:
        parser.error('context, horizon and cycle_step must be positive')
    if not isinstance(args.selection_metric, str) or not args.selection_metric:
        parser.error('selection_metric must be loss or task.metric')
    if args.aggregation == 'cell_macro' and args.domain != 'cell':
        parser.error('cell_macro requires cell samples with cell_id')
    if args.sampling == 'cell_balanced' and args.domain != 'cell':
        parser.error('cell_balanced sampling requires cell data')
    if args.domain == 'cell':
        if not 0 < args.dataaccess <= 100:
            parser.error('dataaccess must be in (0, 100]')
        if args.eval_cycle_min is not None and args.eval_cycle_max is not None and args.eval_cycle_min > args.eval_cycle_max:
            parser.error('eval_cycle_min must not exceed eval_cycle_max')
    elif (not 0 < args.val_fraction < 1 or not 0 < args.test_fraction < 1
          or args.val_fraction + args.test_fraction >= 1):
        parser.error('Validation/test fractions must be positive and sum to less than one')
    elif args.dataaccess != 100 or args.condition != 'all' or args.eval_cycle_min is not None or args.eval_cycle_max is not None:
        parser.error('dataaccess, condition and eval_cycle filters are supported only for cell data')
    for value in (args.run_name, args.dataset, args.model, args.condition):
        if value is not None and (not value or Path(value).name != value or value in ('.', '..')):
            parser.error('Run name, dataset, model and condition must be single path components')
    return args


def setting_name(args, repeat=0):
    config = {key: value for key, value in vars(args).items()
              if key not in ('config', 'checkpoint', 'checkpoints', 'output_dir', 'run_name', 'is_training', 'stage', 'itr',
                             'use_gpu', 'gpu', 'devices', 'num_workers')}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    name = args.run_name or f'{args.domain}_{args.model}_{"-".join(args.tasks)}_{digest}'
    return f'{name}_repeat{repeat}' if args.itr > 1 else name
