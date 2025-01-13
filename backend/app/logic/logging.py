import logging
from enum import Enum


class CustomFormatter(logging.Formatter):

    grey: str = "\x1b[38;20m"
    cyan: str = "\x1b[36;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = "%(levelname)-8s (%(filename)-26s:%(lineno)3d:%(funcName)-30s) - %(message)s "

    FORMATS: dict[int, str] = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord):  # pyright: ignore [reportImplicitOverride]
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class LogLevel(Enum):
    NOTSET = logging.NOTSET
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    DEBUG = logging.DEBUG
    INFO = logging.INFO


def get_logger(name: str, log_level: LogLevel) -> logging.Logger:
    logger = logging.getLogger(name)
    logger_stream = logging.StreamHandler()
    logger_stream.setLevel(log_level.value)
    # formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
    logger_stream.setFormatter(CustomFormatter())
    logger.setLevel(log_level.value)
    logger.addHandler(logger_stream)
    logger.propagate = False
    return logger
