"""
Backwards-compatible entry point for training the propagation
(graph + temporal + text + linguistic) model.

This used to be a second, independent training loop that duplicated
(and partially diverged from) ``scripts/train.py``: it had no LR
warmup/schedule, no early stopping, no AMP, and always used an
unstratified-in-name-only split copied by hand. Maintaining two
copies of the training loop is exactly how bugs like that creep in
and go unnoticed, so this script now just forwards to the single,
shared, fixed pipeline in ``scripts/train.py`` with ``--model
propagation`` forced.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import main as train_main


def main():
    argv = sys.argv[1:]

    # Map the old --lr flag (if present) onto the new
    # --backbone-lr/--head-lr split for backwards compatibility.
    if "--lr" in argv:
        idx = argv.index("--lr")
        lr_value = argv[idx + 1]

        del argv[idx: idx + 2]

        argv += ["--backbone-lr", lr_value]

    argv = ["--model", "propagation"] + argv

    train_main(argv)


if __name__ == "__main__":
    main()
