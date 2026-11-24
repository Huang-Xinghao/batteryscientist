"""Contract for adding prediction domains without copying the experiment loop."""

from abc import ABC, abstractmethod


class PredictionAdapter(ABC):
    # forward returns {task: (prediction, scaled_target, boolean_observed_mask)}.
    # decode converts prediction/target to original evaluation units.
    # state must be JSON-serializable and include split manifests and all scalers.
    @abstractmethod
    def loaders(self, state=None):
        """Build train/validation/test, restoring preprocessing when state is supplied."""

    @abstractmethod
    def state(self):
        """Return preprocessing, data provenance and split membership."""

    @abstractmethod
    def forward(self, model, batch, device):
        """Adapt model outputs and observed task targets for shared loss computation."""

    @abstractmethod
    def decode(self, task, prediction, target, batch):
        """Convert standardized arrays to original task units."""

    @abstractmethod
    def samples(self, batch):
        """Return stable sample identifiers in batch order."""

    def loss_parts(self, task, prediction, target, mask):
        """Override for research losses; return a differentiable sum and its weight."""
        from torch.nn import functional as F
        functions = {'MSE': F.mse_loss, 'MAE': F.l1_loss, 'Huber': F.smooth_l1_loss}
        return functions[getattr(self.args, 'loss', 'MSE')](
            prediction[mask], target[mask], reduction='sum'), int(mask.sum())

    def optimizer(self, model):
        from torch.optim import Adam
        return Adam(model.parameters(), lr=self.args.learning_rate)

    def metrics(self, task, prediction, target, observed, samples):
        from evaluation import regression_metrics
        return regression_metrics(prediction, target, observed, samples,
                                  getattr(self.args, 'aggregation', 'observation'))

    def extras(self, decoded, batch):
        """Optional derived outputs saved alongside task predictions."""
        return {}
