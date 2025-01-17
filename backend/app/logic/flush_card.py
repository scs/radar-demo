from app.logic.config import STATIC_CONFIG
from app.logic.logging import LogLevel, get_logger

logger = get_logger(__name__, LogLevel.ERROR)


def flush_card(timeout_ms: int) -> bool:
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.flush(timeout_ms)
        if err != 0:
            eib(LogLevel.ERROR)
            eob(LogLevel.ERROR)
            oib(LogLevel.ERROR)
            oob(LogLevel.ERROR)
        return err == 0
    return True


def buffer_status(level: LogLevel) -> None:
    eib(level)
    eob(level)
    oib(level)
    oob(level)


def eib(level: LogLevel = LogLevel.NOTSET) -> None:
    if STATIC_CONFIG.versal_lib:
        val: int = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
        log(val, "empty input buffers", level)


def eob(level: LogLevel = LogLevel.NOTSET) -> None:
    if STATIC_CONFIG.versal_lib:
        val: int = STATIC_CONFIG.versal_lib.num_empty_output_buffers()
        log(val, "empty output buffers", level)


def oib(level: LogLevel = LogLevel.NOTSET) -> None:
    if STATIC_CONFIG.versal_lib:
        val: int = STATIC_CONFIG.versal_lib.num_occupied_input_buffers()
        log(val, "occupied input buffers", level)


def oob(level: LogLevel = LogLevel.NOTSET) -> None:
    if STATIC_CONFIG.versal_lib:
        val: int = STATIC_CONFIG.versal_lib.num_occupied_output_buffers()
        log(val, "occupied output buffers", level)


def log(val: int, message: str, level: LogLevel) -> None:
    match level:
        case LogLevel.DEBUG:
            logger.debug(f"There are {val} {message}")
        case LogLevel.INFO:
            logger.info(f"There are {val} {message}")
        case LogLevel.WARNING:
            logger.warning(f"There are {val} {message}")
        case LogLevel.ERROR:
            logger.error(f"There are {val} {message}")
        case _:
            ...
