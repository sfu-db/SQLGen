from typing import Union  # NOQA

from sqlgen.storages.base import BaseStorage  # NOQA
from sqlgen.storages.in_memory import InMemoryStorage


def get_storage(storage=None):
    # type: (Union[None, str, BaseStorage]) -> BaseStorage

    if storage is None:
        return InMemoryStorage()
    else:
        return storage
