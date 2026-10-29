"""Shared CHM model selection and strict checkpoint loading for both toolboxes.

Load sibling files by absolute path: ArcGIS need not add this folder to sys.path.
The legacy U-Net class is supplied by the toolbox to preserve existing weights.
"""

import importlib.util
import json
from pathlib import Path

import torch

MODEL_TYPES = ("UNet", "ResUNet", "UNet++")
_MODEL_FILES = {
    "ResUNet": ("resunet.py", "ResUNetRegressor"),
    "UNet++": ("unetplusplus.py", "UNetPlusPlusRegressor"),
}


def model_config(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown CHM model {model_type!r}. Choose one of {MODEL_TYPES}.")
    return {
        "architecture_version": 1,
        "in_channels": 4,
        "out_channels": 1,
        "base_channels": 64 if model_type == "UNet" else 32,
        "output_activation": "ReLU",
    }


def build_chm_model(model_type, legacy_unet_class):
    config = model_config(model_type)
    if model_type == "UNet":
        return legacy_unet_class(in_channels=4, out_channels=1)
    filename, class_name = _MODEL_FILES[model_type]
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location("_chm_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name)
    return model_class(in_channels=4, out_channels=1, base_channels=config["base_channels"])


def load_chm_weights(model, weights_path, model_type):
    """Check metadata, tensor names and shapes before applying any weights."""
    config = model_config(model_type)
    metadata_path = Path(weights_path).with_suffix(".metadata.json")
    if metadata_path.is_file():
        with metadata_path.open(encoding="utf-8-sig") as stream:
            metadata = json.load(stream)
        if (metadata.get("model_type") != model_type or metadata.get("input_channels") != 4
                or metadata.get("output_channels") != 1):
            raise ValueError(f"Checkpoint metadata does not match the selected four-band CHM {model_type}. "
                             "Select the Model Type used to train these weights.")
        if metadata.get("preprocessing", {}).get("normalization_divisor") != 255.0:
            raise ValueError("Checkpoint metadata must use the fixed normalization divisor of 255.")
        if "model_config" in metadata and metadata["model_config"] != config:
            raise ValueError("Checkpoint architecture configuration does not match this model version.")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("The .pth file must contain a PyTorch state dictionary.")
    state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint has no model state dictionary.")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("State dictionary must map tensor names to tensors.")
        name = key.removeprefix("module.")
        if name in cleaned:
            raise ValueError(f"Duplicate checkpoint tensor name: {name}")
        cleaned[name] = value
    expected = model.state_dict()
    if set(cleaned) != set(expected) or any(cleaned[name].shape != expected[name].shape
                                           for name in expected if name in cleaned):
        raise ValueError(f"Weights do not match the selected four-band CHM {model_type}. "
                         "All tensor names and shapes must match.")
    model.load_state_dict(cleaned, strict=True)
    return len(cleaned)
