from typing import Iterable, Optional, Callable, Any, TypeVar
import ctypes
from sklearn.metrics.pairwise import cosine_similarity

_T = TypeVar('_T')
class TokenSequence(list[_T]):


    def extend(self, iterable: Iterable[_T]) -> None:
        super().extend(iterable)
        
    def insert(self, index: int, object: _T) -> None:
        super().insert(index, object)

    def append(self, object: _T) -> None:
        super().append(object)

    def remove(self, object: _T) -> None:
        super().remove(object)

    def pop(self, index: int = ...) -> _T:
        return super().pop(index)

    def sort(self, *, key: Optional[Callable[[_T], Any]] = ..., reverse: bool = ...) -> None:
        super().sort(key=key, reverse=reverse)

    def clear(self) -> None:
        super().clear()

    def reverse(self) -> None:
        super().reverse()

    def __hash__(self):
        value = 0x345678
        for item in self:
            value = c_mul(1000003, value) ^ hash(item)
        value = value ^ len(self)
        if value == -1:
            value = -2
        return value


def c_mul(a, b):
    # return eval(hex((long(a) * b) & 0xFFFFFFFFL)[:-1])
    return ctypes.c_int64((a * b) & 0xffffffff).value