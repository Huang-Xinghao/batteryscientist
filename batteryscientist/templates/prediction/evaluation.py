"""Metric aggregation is independent of model inputs and training loss."""
import numpy as np


def regression_metrics(prediction, target, observed, samples, aggregation='observation'):
    def scores(rows):
        mask = observed[rows]
        error = (prediction[rows] - target[rows])[mask]
        if not len(error):
            raise ValueError('No observed values in evaluation group')
        nonzero = mask & (np.abs(target[rows]) > 1e-8)
        mape = np.abs((prediction[rows][nonzero] - target[rows][nonzero]) / target[rows][nonzero])
        return dict(mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                    mape_percent=float(mape.mean()*100) if len(mape) else None)
    if aggregation == 'observation':
        return dict(scores(slice(None)), aggregation=aggregation)
    if aggregation != 'cell_macro':
        raise ValueError(f'Unknown aggregation {aggregation!r}')
    groups = {}
    for i, sample in enumerate(samples):
        groups.setdefault(sample['cell_id'], []).append(i)
    per_cell = {cell: scores(rows) for cell, rows in groups.items() if observed[rows].any()}
    if not per_cell:
        raise ValueError('No observed cells')
    result = {key: float(np.mean([value[key] for value in per_cell.values() if value[key] is not None]))
              if any(value[key] is not None for value in per_cell.values()) else None
              for key in ('mae', 'rmse', 'mape_percent')}
    return dict(result, aggregation=aggregation, cells=len(per_cell), per_cell=per_cell)
