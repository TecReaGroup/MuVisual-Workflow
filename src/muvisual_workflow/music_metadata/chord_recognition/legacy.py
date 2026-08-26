"""Compatibility runner for a locally installed legacy Chord-CNN-LSTM script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def main() -> None:
    script_path = Path(sys.argv[1]).resolve()
    for name, value in (("bool", bool), ("complex", complex), ("float", float), ("int", int), ("object", object), ("str", str)):
        if name not in np.__dict__:
            setattr(np, name, value)

    original_yaml_load = yaml.load

    def yaml_load(stream: object, *args: object, **kwargs: object) -> object:
        if not args and "Loader" not in kwargs:
            kwargs["Loader"] = yaml.SafeLoader
        return original_yaml_load(stream, *args, **kwargs)

    yaml.load = yaml_load
    original_torch_load = torch.load

    def torch_load(source: object, *args: object, **kwargs: object) -> object:
        if not torch.cuda.is_available() and "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return original_torch_load(source, *args, **kwargs)

    torch.load = torch_load
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
