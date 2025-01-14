import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("radar demo app")


class Timer:
    DEBUG: bool = False

    def __init__(self, name: str):
        self.name: str = name
        self.start_time: float = time.time()

    def start(self):
        self.start_time = time.time()

    def duration(self) -> float:
        stop = time.time()
        duration = stop - self.start_time
        self.start_time = time.time()
        return duration

    def snapshot(self) -> float:
        return time.time() - self.start_time

    def log_time(self) -> None:
        if Timer.DEBUG:
            logger.info(self)

    def __str__(self):  # pyright: ignore [reportImplicitOverride]
        duration = self.duration()
        return f"Timer {self.name} {duration} sec := {1/duration} fps"
