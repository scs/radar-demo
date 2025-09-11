from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class Functor(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value

    def bind(self, func: Callable[[T], U]) -> Functor[U]:
        return Functor(func(self.value))
