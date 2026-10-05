"""
Model Registry

Authors: LIDARLearn contributors
License: See repository LICENSE

Defines the global MODELS registry used by every @MODELS.register_module()
class and the build_model_from_cfg helper consumed by the runners.
"""
from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT):
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(cfg, **kwargs)
