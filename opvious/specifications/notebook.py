import logging
import os
import types
from typing import Optional
import warnings

from .model import Model


_logger = logging.getLogger(__name__)


def load_notebook_models(
    path: str, root: Optional[str] = None
) -> types.SimpleNamespace:
    """Loads all models from a notebook

    Args:
        path: Path to the notebook, relative to `root` if present otherwise CWD
        root: Root path. If set to a file, its parent directory will be used
            (convenient for use with `__file__`).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=DeprecationWarning,
            module=r".*(importnb|IPython)",
        )
        import importnb  # type: ignore

    if root:
        root = os.path.realpath(root)
        if os.path.isfile(root):
            root = os.path.dirname(root)
        path = os.path.join(root, path)
    nb = importnb.Notebook.load_file(path)

    ns = types.SimpleNamespace()
    count = 0
    for attr in dir(nb):
        value = getattr(nb, attr)
        if isinstance(value, Model):
            count += 1
            setattr(ns, attr, value)
    if not count:
        raise Exception("No models found")
    _logger.debug("Loaded %s model(s) from %s.", count, path)

    return ns
