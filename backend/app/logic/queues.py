import queue
from collections.abc import Generator
from typing import Generic, TypeVar, final

T = TypeVar("T")


@final
class QueueList(Generic[T]):
    def __init__(self, num_queues: int, maxsize: int = 0):
        self.queues = [queue.Queue(maxsize) for _ in range(num_queues)]

    def flush(self):
        for q in self.queues:
            while not q.empty():
                _ = q.get()

    def anyfull(self) -> bool:
        for q in self.queues:
            if q.full():
                return True
        return False

    def anyempty(self) -> bool:
        for q in self.queues:
            if q.empty():
                return True
        return False

    def __getitem__(self, idx: int) -> queue.Queue[T]:
        return self.queues[idx]

    def __iter__(self) -> Generator[queue.Queue[T], None, None]:
        for q in self.queues:
            yield q


result_queues = QueueList(num_queues=4, maxsize=2)
receive_queues = QueueList(num_queues=4, maxsize=2)
target_queues = QueueList(num_queues=4, maxsize=2)
