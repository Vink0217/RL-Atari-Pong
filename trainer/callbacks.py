import os
import glob
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# -----------------------------------------------------------------
# This is your existing callback. It's good, we keep it.
# -----------------------------------------------------------------
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
        last = getattr(self.eval_callback, "last_mean_reward", None)
        if last is None or last == -float("inf"):
            return True

        if last > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = last
            self.counter = 0
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

# -----------------------------------------------------------------
# This is the NEW callback that replaces CheckpointCallback
# -----------------------------------------------------------------
class RotatingCheckpointCallback(CheckpointCallback):
    """
    A CheckpointCallback that deletes old checkpoints to keep only the 'keep_last' most recent ones.
    """
    def __init__(self, keep_last: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        # First, run the normal CheckpointCallback logic
        # This will save a new model if it's time
        super()._on_step()

        # Now, add our custom rotation logic
        checkpoints = self._get_checkpoint_files()
        
        # If we have more checkpoints than we want to keep
        if len(checkpoints) > self.keep_last:
            # Get the list of checkpoints to delete (all but the most recent 'keep_last')
            checkpoints_to_delete = checkpoints[:-self.keep_last]
            
            if self.verbose > 0:
                print(f"[RotatingCheckpointCallback] Deleting {len(checkpoints_to_delete)} old checkpoints...")
            
            for f in checkpoints_to_delete:
                try:
                    os.remove(f)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"[RotatingCheckpointCallback] Error deleting {f}: {e}")

        return True

    def _get_checkpoint_files(self) -> list:
        """Helper to get a sorted list of checkpoint files."""
        # Find all files in the save path that match the prefix
        file_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
        checkpoints = glob.glob(file_pattern)
        
        # Sort them by the number of steps (most recent last)
        checkpoints.sort(key=lambda f: int(f.split('_')[-2]))
        
        return checkpoints