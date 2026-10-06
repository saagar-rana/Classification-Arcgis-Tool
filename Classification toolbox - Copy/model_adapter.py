"""Load selected research models without importing the aggregate models package."""

import importlib.util
import os
import sys
import types
from types import SimpleNamespace


def load_local_model(project_dir, package_name, class_name):
    """Load a sibling model package by path, independent of ArcGIS sys.path."""
    package_dir = os.path.join(project_dir, package_name)
    source_path = os.path.join(package_dir, package_name + ".py")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Model source not found: {source_path}")
    namespace = "_quebec_toolbox_" + package_name
    module_name = namespace + "." + package_name
    if module_name not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [package_dir]
        sys.modules[namespace] = package
        try:
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    sys.modules.pop(name, None)
            raise
    return getattr(sys.modules[module_name], class_name)


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


def build_dgcnn(models_dir, num_classes):
    config = SimpleNamespace(num_classes=num_classes, channels_input=7,
                             k=20, emb_dims=1024, dropout=0.3)
    return load_dgcnn(models_dir)(config=config)
