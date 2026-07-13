"""InterWeave controller training entry point.

This wrapper keeps the paper-facing command names aligned with InterWeave
Reasoning while reusing the original PPO training implementation.
"""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    run_path(str(Path(__file__).with_name("train_ppo_controller.py")), run_name="__main__")
