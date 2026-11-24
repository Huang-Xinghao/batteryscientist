"""Unified entry point for cell and electrolyte supervised prediction."""

import random
from copy import deepcopy

import numpy as np
import torch

from configuration import parse_args, setting_name
from exp.experiment import Experiment


def main(argv=None):
    args = parse_args(argv)
    for repeat in range(args.itr):
        seed = args.seed + repeat
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        setting = setting_name(args, repeat)
        run_args = deepcopy(args)
        run_args.seed = seed
        experiment = Experiment(run_args)
        if args.stage == 'check':
            experiment.check(setting)
        elif args.stage in ('train', 'train_test'):
            print(f'Start training: {setting}')
            experiment.train(setting)
            experiment.validate(setting)
            if args.stage == 'train_test':
                experiment.test(setting)
        elif args.stage == 'validate':
            experiment.validate(setting, restore=True, checkpoint=args.checkpoint)
        else:
            experiment.test(setting, test=1, checkpoint=args.checkpoint)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
