from app.logic.config import STATIC_CONFIG
from app.logic.logging import LogLevel, get_logger

logger = get_logger(__name__, LogLevel.ERROR)


def flush_card(timeout_ms: int) -> bool:
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.flush(timeout_ms)
        if err != 0:
            logger.error("Unable to flush the card()")
            eib: int = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
            eob: int = STATIC_CONFIG.versal_lib.num_empty_output_buffers()
            oib: int = STATIC_CONFIG.versal_lib.num_occupied_input_buffers()
            oob: int = STATIC_CONFIG.versal_lib.num_occupied_output_buffers()
            logger.error(f"There are {eib} empty input buffers")
            logger.error(f"There are {eob} empty output buffers")
            logger.error(f"There are {oib} occupied input buffers")
            logger.error(f"There are {oob} occupied output buffers")
        return err == 0
    return True
