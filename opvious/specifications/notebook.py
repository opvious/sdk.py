import os
from typing import Optional
import warnings

from .local import LocalSpecification
from .model import Model


def load_notebook_specification(
    path: str, attr: Optional[str] = None, root: Optional[str] = None
) -> LocalSpecification:
    """Loads a local specification from a notebook"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            category=DeprecationWarning,
            module=r'.*(importnb|IPython)'
        )
        import importnb  # type: ignore

    if root:
        root = os.path.realpath(root)
        if os.path.isfile(root):
            root = os.path.dirname(root)
        path = os.path.join(root, path)
    nb = importnb.Notebook.load_file(path)

    model = None
    if attr is None:
        for attr in dir(nb):
            value = getattr(nb, attr)
            if isinstance(value, Model):
                if model:
                    raise Exception(
                        "Multiple models found, please disambiguate"
                    )
                model = value
    else:
        model = getattr(nb, attr)
    if not model:
        raise Exception("No model found")

    sources = model.compile_specification_sources()
    # TODO: Set description from notebook markdown cells
    return LocalSpecification(sources=sources, description=model.__doc__)
