"""InterWeave code-generation evaluation entry point."""

from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    run_path(str(Path(__file__).with_name("eval_code_ppo_agent.py")), run_name="__main__")
