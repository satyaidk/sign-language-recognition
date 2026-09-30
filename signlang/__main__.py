"""
Command-line entry point:  python -m signlang <command> [options]

Each command maps to one pipeline stage; ``python -m signlang <command> --help``
shows its options.  Stage modules are imported lazily, so e.g. ``train`` works
without MediaPipe installed and ``--help`` is instant.
"""
from __future__ import annotations

import importlib
import sys

from signlang import __version__
from signlang.utils import enable_utf8_console

# command -> (module, one-line description), in pipeline order
COMMANDS = {
    "view":     ("signlang.landmarks.viewer",  "L0  live landmark viewer (webcam / video / image)"),
    "extract":  ("signlang.dataset.extract",   "L1  video dataset -> landmark dataset (+ skeletons, QA)"),
    "verify":   ("signlang.dataset.verify",    "L1  cross-check the saved landmarks (+ overlay videos)"),
    "prune":    ("signlang.dataset.prune",     "L1  reversibly drop clips from the dataset"),
    "report":   ("signlang.dataset.report",    "L1  dataset / verification infographics"),
    "observe":  ("signlang.dataset.observe",   "L1  analyse one long continuous-signing video"),
    "train":    ("signlang.training.train",    "L2  stage 1: k-fold cross-validation"),
    "finetune": ("signlang.training.finetune", "L2  stage 2: fit the final model on all data"),
    "export":   ("signlang.training.export",   "L2  stage 3: export to ONNX (+ parity, latency)"),
    "video":    ("signlang.inference.video",   "L3  stage 4: recognise signs in a video file"),
    "live":     ("signlang.inference.live",    "L3  stage 5: live webcam recognition"),
}


def usage() -> str:
    lines = [f"signlang {__version__} — isolated sign-language recognition from MediaPipe landmarks",
             "", "usage: python -m signlang <command> [options]", "", "commands:"]
    lines += [f"  {name:<9} {desc}" for name, (_, desc) in COMMANDS.items()]
    lines += ["", "Run `python -m signlang <command> --help` for a command's options."]
    return "\n".join(lines)


def main(argv=None) -> int:
    enable_utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    if argv[0] in ("-V", "--version"):
        print(__version__)
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"unknown command: {command}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    module.main(rest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
