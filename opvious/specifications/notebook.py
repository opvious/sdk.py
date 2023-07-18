import logging
import os
import threading
import types
from typing import Optional
import warnings

from ..modeling import Model


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
    if root:
        root = os.path.realpath(root)
        if os.path.isfile(root):
            root = os.path.dirname(root)
        path = os.path.join(root, path)
    ns = types.SimpleNamespace()

    # We run the import logic in a separate, fresh, thread since `importnb`
    # expects an inactive event loop if the notebook includes async statements.
    t = _ImportThread(target=_populate_notebook_namespace, args=(path, ns))
    t.start()
    t.join()

    return ns


class _ImportThread(threading.Thread):
    """Thread which rethrows any import exception"""

    _exception = None

    def run(self):
        try:
            super().run()
        except Exception as e:
            self._exception = e

    def join(self):
        super().join()
        if self._exception:
            raise Exception("Notebook import failed") from self._exception


def _populate_notebook_namespace(path: str, ns: types.SimpleNamespace) -> None:
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=DeprecationWarning,
            module=r".*(importnb|IPython)",
        )
        import importnb  # type: ignore

    class _Notebook(importnb.Notebook):
        def code(self, raw):
            # We skip magic cells (this is done manually since the default
            # `no_magic` option only skips cells starting with 2 %).
            if raw.startswith("%"):
                return "# " + raw
            # We use a custom loader to transform all top-level awaited
            # expressions into statements, otherwise their value will show up
            # in importing notebooks. Note that this isn't fool-proof since we
            # rely on the `await` keyword being first on the line.
            lines = [
                f"_ = {s}" if s.startswith("await ") else s
                for s in raw.split("\n")
            ]
            return super().code("\n".join(lines))

    nb = _Notebook.load_file(path)

    count = 0
    for attr in dir(nb):
        value = getattr(nb, attr)
        if isinstance(value, Model):
            count += 1
            setattr(ns, attr, value)
    if not count:
        raise Exception("No models found")
    _logger.debug("Loaded %s model(s) from %s.", count, path)
