import ctypes
import hashlib
import threading
import time
from ctypes import cdll
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from app.logic.flush_card import buffer_status, eib, eob, oib, oob
from app.logic.logging import LogLevel, get_logger
from app.logic.output_exception import InputFull, OutputEmpty
from app.logic.timer import Timer
from app.package import package_root

logger = get_logger(__name__, LogLevel.DEBUG)
ARTEFACTS = package_root / "artefacts"
DATA_SERVER = ARTEFACTS / "libdata_server.so"
EXPORT = False


class HwError(Exception): ...


def probe_hw():
    class LibOption(Enum):
        HW = 0
        EMULATE = 1

    lib_option = LibOption.HW.value

    if Path.exists(DATA_SERVER):
        try:
            versal_lib = cdll.LoadLibrary(str(DATA_SERVER))
            if versal_lib:
                return_code: int = versal_lib.init_server(lib_option)
                if return_code != lib_option:
                    logger.error(f"Failed to open Shared Object return code was {return_code} expected {lib_option}")
                    raise HwError
                return versal_lib
        except Exception:
            raise HwError

    raise HwError


def get_steps():
    steps = np.array([0, 0, 0, 0])
    while True:
        yield list(steps)
        steps += 1
        steps %= [90, 120, 180, 320]


steps = get_steps()
versal_lib = probe_hw()
send_count: threading.Semaphore = threading.Semaphore(0)
producer_run = threading.Event()
receiver_run = threading.Event()
receiver_capture = threading.Event()


def send_scene(timeout_ms: float, frame_nr: int) -> int:
    num_channels: int = 4
    step: list[int] = next(steps)
    timeout = Timer("send_timeout")
    for idx in range(4):
        timeout.start()  # each radar has the same timeout
        while not versal_lib.input_ready():
            if timeout.snapshot() / 1000 > timeout_ms:
                logger.error("Input full timeout")
                raise InputFull()
            else:
                time.sleep(0.01)

        err: int = 500
        for _ in range(3):  # Try max three times to send the same data
            # logger.debug(f"SENT: idx = {idx}, frame_nr = {frame_nr}, step = {step[idx]}, num_channels = {num_channels}")
            err = versal_lib.send_scene(0, int(frame_nr), 0, int(num_channels), 0)
            if err == 0:
                send_count.release()
                break

        if err:
            logger.error("####################################################################")
            logger.error("####################    Unable to send scene    ####################")
            logger.error("####################################################################")
            raise HwError
        else:
            frame_nr += 1

    return frame_nr


def sender():
    timer: Timer = Timer("send_radar_scene")
    frame_nr: int = 0
    major = 0
    while True:
        major += 1
        for i in range(1, 81):
            logger.debug(f"Major iteration {major} Minor iteration {i}")
            for _ in range(i):
                try:
                    frame_nr = send_scene(2 * 60, frame_nr)
                    timer.log_time()
                except InputFull:
                    continue

            timeout = Timer(name="Timeout")
            while send_count._value != 0:
                if timeout.snapshot() > 4:
                    receiver_capture.set()
                    logger.error(f"Set capture event for minor iteration {i} frame number {frame_nr}")
                    time.sleep(1)
                    frame_nr = send_scene(2 * 60, frame_nr)
                    logger.error("Sent flush frame")
                    raise TimeoutError
                time.sleep(0.1)

    logger.info("Producer Stopped")
    receiver_run.clear()


def receive_radar_result() -> tuple[int, int, int, NDArray[np.int16]]:
    complex_result = np.empty((1024, 512), np.int16)
    timer = Timer(name="get_radar_result")
    err = 0
    idx = ctypes.c_uint32(0)
    step = ctypes.c_uint32(0)
    frame_nr = ctypes.c_uint32(0)
    if versal_lib.output_ready():
        err: int = versal_lib.receive_result(
            complex_result.ctypes,
            ctypes.byref(idx),
            ctypes.byref(step),
            ctypes.byref(frame_nr),
            1024 * 512 * ctypes.sizeof(ctypes.c_int16),
            0,
        )
        if err:
            logger.error("###############################################################")
            logger.error("########           Unable to receive result           #########")
            logger.error("###############################################################")
    else:
        # logger.warning("No occupied output buffer available")
        raise OutputEmpty()
    timer.log_time()
    # logger.debug(f"RECEIVED idx = {idx.value}, frame_nr = {frame_nr.value}, step = {step.value}")
    return (
        (idx.value, step.value, frame_nr.value, complex_result)
        if err == 0
        else (0, step.value, frame_nr.value, np.zeros((1024, 512)).astype(np.int16))
    )


def make_check(update: Callable[[int], int]) -> Callable[[int], None]:
    expected = 0

    def check_expected(actual: int) -> None:
        nonlocal expected
        if actual != expected:
            logger.error(f"Expected index {expected}, actual {actual}")
            # raise Exception(f"Expected index {expected[0]}, actual {actual}")
        expected = update(actual)
        # expected = (actual + 1) % get_result_range().stop

    return check_expected


def receiver() -> None:
    check_frame_nr = make_check(lambda x: x + 1)
    check_radar_idx = make_check(lambda x: (x + 1) % 4)
    prev = {"radar_idx": 0, "step": 0, "frame_nr": 0}
    prev_result = np.zeros((1024, 512)).astype(np.int16)
    while receiver_run.is_set():
        try:
            radar_idx, step, frame_nr, result = receive_radar_result()
            _ = send_count.acquire()
            export_ref(result, step, radar_idx)
            hash_result(result, step, radar_idx, frame_nr)
            # if receiver_capture.is_set():
            #     export_inspect(prev_result, prev["step"], prev["radar_idx"], prev["frame_nr"])
            #     export_inspect(result, step, radar_idx, frame_nr)
            # if step != 0:
            #     logger.error("Step should always be 0")

            prev["radar_idx"] = radar_idx
            prev["step"] = step
            prev["frame_nr"] = frame_nr
            prev_result = result

            # check_radar_idx(radar_idx)
            check_frame_nr(frame_nr)
        except OutputEmpty:
            time.sleep(0.008)  # wait for 8 ms (half the time that one cycle should take)

    log_timer = Timer(name="LogTim")
    while send_count.acquire(blocking=False):
        received = False
        while not received:
            if log_timer.snapshot() > 2:
                log_timer.start()
                eib(LogLevel.ERROR)
                eob(LogLevel.ERROR)
                oib(LogLevel.ERROR)
                oob(LogLevel.ERROR)
            try:
                _, _, _, _ = receive_radar_result()
                received = True
            except OutputEmpty:
                time.sleep(0.01)

    buffer_status(LogLevel.INFO)
    logger.info("Receiver Stopped")


def start_threads() -> None:
    global sender_thread
    global receiver_thread
    producer_run.set()
    receiver_run.set()
    receiver_capture.clear()
    sender_thread = threading.Thread(target=sender, name="producer")
    sender_thread.start()
    receiver_thread = threading.Thread(target=receiver, name="consumer")
    receiver_thread.start()


def test_flush() -> None:
    start_threads()


def export_ref(intensity_image: NDArray[np.int16], step: int, radar_idx: int) -> None:
    if EXPORT:
        result_file_name = f"results/result_channel_{radar_idx}_position_{int(step):04d}.bin"
        result_file_path = Path(result_file_name)
        if not result_file_path.is_file():
            intensity_image.tofile(result_file_path)


md5_dict = {0: {}, 1: {}, 2: {}, 3: {}}


def hash_result(intensity_image: NDArray[np.int16], step: int, radar_idx: int, frame_nr: int) -> None:
    # timer = Timer(name="Md5Sum")
    md5 = np.sum(intensity_image)
    ref = md5_dict[radar_idx].setdefault(step, md5)
    if ref != md5:
        logger.error(f"MD5 error for frame {frame_nr}, radar idx = {radar_idx}, step = {step}, sum = {md5}")
        export_inspect(intensity_image, step, radar_idx, frame_nr)
    if step == 0:
        export_ref(intensity_image, step, radar_idx)



def export_inspect(intensity_image: NDArray[np.int16], step: int, radar_idx: int, frame_nr: int) -> None:
    logger.error(f"Exporting for inspection {frame_nr}")
    result_file_name = f"inspect/result_channel_{radar_idx}_position_{int(step):04d}_{int(frame_nr)}.bin"
    result_file_path = Path(result_file_name)
    if not result_file_path.is_file():
        intensity_image.tofile(result_file_path)


if __name__ == "__main__":
    test_flush()
