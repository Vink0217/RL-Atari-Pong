import os
from stable_baselines3.common.callbacks import BaseCallback

class EarlyStopOnReward(BaseCallback):
    """Stop training when evaluation reward hasn't improved for `patience` evaluations."""


    def __init__(self, eval_callback, patience=5, min_delta=0.0, verbose=1):
        super().__init__()
        self.eval_callback = eval_callback
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.counter = 0
        self.verbose = verbose


    def _on_step(self) -> bool:
    # The eval callback stores last_mean_reward on its object when it runs
        last = getattr(self.eval_callback, "last_mean_reward", None)
        # Some EvalCallback implementations set last_mean_reward to -inf when no
        # episodes were completed during evaluation. Treat -inf the same as None
        # (i.e. skip this check until a finite reward appears).
        if last is None or last == -float("inf"):
            return True

        if last > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = last
            self.counter = 0
            # save a best model copy if available
            try:
                best_path = os.path.join(self.logger.get_dir(), "best_model.zip")
                self.model.save(best_path)
            except Exception:
                pass
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopOnReward: no improvement ({self.counter}/{self.patience}) - best={self.best_mean_reward} last={last}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered - stopping training")
                return False
        return True