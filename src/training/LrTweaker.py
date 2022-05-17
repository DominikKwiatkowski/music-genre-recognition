import math
from tensorflow.keras import backend as K

from training.training_config import TrainingSetup


class LrTweaker:
    """Component for adjusting the learning rate of a model."""

    def __init__(
        self,
        training_config: TrainingSetup,
        patience: int,
        decrease_multiplier: float,
        min_lr: float,
    ):
        self.training_config: TrainingSetup = training_config
        self.patience: int = patience
        self.decrease_multiplier: float = decrease_multiplier
        self.min_lr: float = min_lr

        self.best: float = math.inf
        self.curr_wait: int = 0

    def on_epoch_end(self, loss: float) -> None:
        if loss < self.best:
            self.best = loss
            self.curr_wait = 0
        else:
            self.curr_wait += 1

        if self.curr_wait >= self.patience:
            self.curr_wait = 0
            old_lr = self.training_config.p.learning_rate
            new_lr = max(old_lr * self.decrease_multiplier, self.min_lr)

            self.training_config.p.learning_rate = new_lr
            K.set_value(self.training_config.model.optimizer.learning_rate, new_lr)
