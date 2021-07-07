import importlib
import types
from typing import Any

from sqlgen import distributions  # NOQA
from sqlgen import exceptions  # NOQA
from sqlgen import importance  # NOQA
from sqlgen import logging  # NOQA
from sqlgen import pruners  # NOQA
from sqlgen import samplers  # NOQA
from sqlgen import storages  # NOQA
from sqlgen import study  # NOQA
from sqlgen import trial  # NOQA
from sqlgen import visualization  # NOQA

from sqlgen.exceptions import TrialPruned  # NOQA
from sqlgen.study import create_study  # NOQA
from sqlgen.study import delete_study  # NOQA
from sqlgen.study import get_all_study_summaries  # NOQA
from sqlgen.study import load_study  # NOQA
from sqlgen.study import Study  # NOQA
from sqlgen.trial import Trial  # NOQA


class _LazyImport(types.ModuleType):
    """Module wrapper for lazy import.

    This class wraps specified module and lazily import it when they are actually accessed.
    Otherwise, `import optuna` becomes slower because it imports all submodules and
    their dependencies (e.g., bokeh) all at once.
    Within this project's usage, importlib override this module's attribute on the first
    access and the imported submodule is directly accessed from the second access.

    Args:
        name: Name of module to apply lazy import.
    """

    def __init__(self, name: str) -> None:
        super(_LazyImport, self).__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)

structs = _LazyImport("optuna.structs")
