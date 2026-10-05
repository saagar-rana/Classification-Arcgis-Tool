"""Load selected research models without importing the aggregate models package."""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

#sagar
def load_dgcnn(models_dir): 
    """Load the original DGCNN and its relative imports in a private namespace. 

    The toolbox constructs models directly, so only the registration decorator
    is needed from the research framework's otherwise unavailable registry.
    """
    namespace = "_quebec_toolbox_models"
    module_name = namespace + ".dgcnn.dgcnn"
    if module_name in sys.modules:
        return sys.modules[module_name].DGCNN

    created = []
    try:
        for name, path in ((namespace, models_dir),
                           (namespace + ".dgcnn", os.path.join(models_dir, "dgcnn"))):
            package = types.ModuleType(name)
            package.__path__ = [path]
            sys.modules[name] = package
            created.append(name)

        class RegistrationOnly:
            def register_module(self):
                return lambda model_class: model_class

        build = types.ModuleType(namespace + ".build")
        build.MODELS = RegistrationOnly()
        sys.modules[build.__name__] = build
        created.append(build.__name__)

        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(models_dir, "dgcnn", "dgcnn.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        created.append(module_name)
        spec.loader.exec_module(module)
        return module.DGCNN
    except Exception:
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name, None)
        raise


def build_dgcnn(models_dir , num_classes):
    config = SimpleNamespace(num_classes=num_classes, channels_input=7,
                             k=20, emb_dims=1024, dropout=0.3)
    return load_dgcnn(models_dir)(config=config)
