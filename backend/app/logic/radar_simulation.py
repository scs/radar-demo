from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from app.logic.buffer_status import buffer_status
from app.logic.cfar import cfar, sw_cfar
from app.logic.config import STATIC_CONFIG
from app.logic.ctypes_data_blob import AoAEntry, DataBlob, DopplerRangeEntry
from app.logic.functor import Functor
from app.logic.image_utils import create_frame, heat_map, norm_image
from app.logic.logging import LogLevel, get_logger
from app.logic.model import Model
from app.logic.output_exception import InputFull, OutputEmpty
from app.logic.queues import receive_queues, result_queues, target_queues
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

logger = get_logger(__name__, LogLevel.WARNING)


producer_run = threading.Event()
receiver_run = threading.Event()
converter_run = threading.Event()

# locks to make sure only one of the multiple gen_frames can start / stop the threads
stop_lock: threading.Lock = threading.Lock()
start_lock: threading.Lock = threading.Lock()

send_count: threading.Semaphore = threading.Semaphore(0)

gen_frames_state = [threading.Event(), threading.Event(), threading.Event(), threading.Event()]


#
#######################################################################################################################


def flush_queues() -> None:
    receive_queues.flush()
    result_queues.flush()
    target_queues.flush()
    # _ = flush_card(400)


def start_threads() -> None:
    global sender_thread
    global receiver_thread
    global converter_thread
    if start_lock.acquire(blocking=False):
        producer_run.set()
        receiver_run.set()
        converter_run.set()
        flush_queues()
        sender_thread = threading.Thread(target=sender, name="producer")
        sender_thread.start()
        receiver_thread = threading.Thread(target=receiver, name="consumer")
        receiver_thread.start()
        converter_thread = threading.Thread(target=converter, name="converter")
        converter_thread.start()


def stop_threads() -> None:
    _ = stop_lock.acquire()
    if start_lock.locked():
        logger.info("Stop Threads")
        producer_run.clear()
        sender_thread.join()
        receiver_thread.join()
        converter_thread.join()

        range_doppler_info.reset()

        start_lock.release()
    stop_lock.release()


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    while True:
        try:
            frame = result_queues[idx].get_nowait()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            time.sleep(0.001)
            continue


def modified_steps() -> list[int]:
    steps = [i + 5 for i in GlobalState.get_current_steps()]
    steps[0] = steps[0] % 90
    steps[1] = steps[1] % 120
    steps[2] = steps[2] % 180
    steps[3] = steps[3] % 380
    return steps


def send_scene(timeout_ms: float, frame_nr: int) -> int:
    num_channels = 16 if GlobalState.model == Model.IMAGING else 4
    step: list[int] = modified_steps()
    timeout = Timer("send_timeout")
    for idx in get_result_range():
        timeout.start()  # each radar has the same timeout
        if STATIC_CONFIG.versal_lib:
            while not STATIC_CONFIG.versal_lib.input_ready():
                if timeout.snapshot() / 1000 > timeout_ms:
                    logger.error("Input full timeout")
                    raise InputFull()
                else:
                    time.sleep(0.01)

            err = 500
            for _ in range(3):  # Try max three times to send the same data
                err = STATIC_CONFIG.versal_lib.send_scene(  # pyright: ignore [reportAny]
                    idx, frame_nr, step[idx], num_channels, 0
                )
                if err == 0:
                    send_count.release()
                    break

            if err:
                logger.error("####################################################################")
                logger.error("####################    Unable to send scene    ####################")
                logger.error("####################################################################")
            else:
                frame_nr += 1

    return frame_nr


def sender():
    timer: Timer = Timer("send_radar_scene")
    frame_nr: int = 0
    while producer_run.is_set():
        if GlobalState.use_hw() and GlobalState.is_running():
            try:
                frame_nr = send_scene(2 * 60, frame_nr)
                timer.log_time()
            except InputFull:
                continue
        else:
            buffer_status(LogLevel.INFO)
            time.sleep(0.1)

    logger.info("Producer Stopped")
    receiver_run.clear()


def receive_radar_result() -> tuple[int, int, int, DataBlob]:
    data: DataBlob = DataBlob()
    # Initialize all data in DataBlob to zero
    _ = ctypes.memset(ctypes.addressof(data), 0, ctypes.sizeof(data))
    # complex_result = np.empty((1024, 512), np.int16)
    timer = Timer(name="get_radar_result")
    err = 0
    idx = ctypes.c_uint32(0)
    step = ctypes.c_uint32(0)
    frame_nr = ctypes.c_uint32(0)
    if STATIC_CONFIG.versal_lib:
        if STATIC_CONFIG.versal_lib.output_ready():
            err: int = STATIC_CONFIG.versal_lib.receive_result(  # pyright: ignore [reportAny]
                ctypes.byref(data),
                ctypes.byref(idx),
                ctypes.byref(step),
                ctypes.byref(frame_nr),
                # fetch 3 MByte of data
                # First MB is doppler range data (1024*512*2 bytes)
                # Second MB is CFAR results
                # Third MB is AOA results
                3 * 1024 * 512 * ctypes.sizeof(ctypes.c_int16),
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
    if err == 0:
        return (idx.value, step.value, frame_nr.value, data)
    else:
        zero = DataBlob()
        _ = ctypes.memset(ctypes.addressof(zero), 0, ctypes.sizeof(zero))
        return (0, step.value, frame_nr.value, zero)


def make_update() -> Callable[[Timer], None]:
    count = 0

    def update_status(timer: Timer) -> None:
        nonlocal count
        INTEGRATION_TIME = 4
        if count % INTEGRATION_TIME == 0:
            range_doppler_info.fps = int(INTEGRATION_TIME / timer.duration() / get_result_range().stop)
        count += 1

    return update_status


def make_enqueue() -> Callable[[int, int, DataBlob], None]:
    previous_step = -1
    commit = True

    def enqueue(radar_idx: int, step: int, data: DataBlob) -> None:
        nonlocal commit
        nonlocal previous_step
        # pre condition
        if radar_idx == 0:
            commit = not receive_queues.anyfull() and step != previous_step
            previous_step = step

        if commit:
            receive_queues[radar_idx].put(data)

    return enqueue


def check_expected_frame_nr(actual: int, expected: list[int]) -> None:
    if actual != expected[0]:
        logger.error(f"Expected frame number {expected[0]}, actual {actual}")
        # raise Exception(f"Expected frame number {expected[0]}, actual {actual}")
    expected[0] = actual + 1


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
    timer = Timer(name="receive loop")
    check_frame_nr = make_check(lambda x: x + 1)
    check_radar_idx = make_check(lambda x: (x + 1) % get_result_range().stop)
    enqueue_received = make_enqueue()
    update_status = make_update()
    while receiver_run.is_set():
        if GlobalState.has_hw():
            try:
                radar_idx, step, frame_nr, data = receive_radar_result()
                _ = send_count.acquire()
                check_radar_idx(radar_idx)
                check_frame_nr(frame_nr)
                enqueue_received(radar_idx, step, data)
                update_status(timer)
            except OutputEmpty:
                time.sleep(0.008)  # wait for 8 ms (half the time that one cycle should take)
        else:
            time.sleep(0.1)

    log_timer = Timer(name="LogTim")
    while send_count.acquire(blocking=False):
        received = False
        while not received:
            if log_timer.snapshot() > 2:
                log_timer.start()
                buffer_status(LogLevel.ERROR)
            try:
                _, _, _, _ = receive_radar_result()
                received = True
            except OutputEmpty:
                time.sleep(0.01)

    buffer_status(LogLevel.INFO)

    logger.info("Receiver Stopped")
    converter_run.clear()


def synthetic_result(current_step: int, channel: int) -> NDArray[np.uint8]:
    timer: Timer = Timer(name="synthetic_result")
    phase: NDArray[np.float32] = (  # pyright: ignore [reportAny]
        2 * np.pi * current_step / STATIC_CONFIG.number_of_steps_in_period[channel]
    )
    ypos: int = int((np.cos(phase + np.pi) * 0.9 + 1) / 2 * 1023)
    xpos: int = 511 + int(
        (np.sin(phase))
        * 480
        / (
            STATIC_CONFIG.period_in_seconds[channel] / (STATIC_CONFIG.period_in_seconds[0] - 0.5)
        )  # pyright: ignore [reportAny]
    )

    Y, X = np.ogrid[: STATIC_CONFIG.video_dim, : STATIC_CONFIG.video_dim]
    dist_from_center = np.sqrt(np.square(X - xpos) + np.square(Y - ypos))
    norm_dist_from_center = dist_from_center / STATIC_CONFIG.video_dim

    norm_intensity_image = np.power(1 - norm_dist_from_center, 24)
    intensity_image = norm_intensity_image * 255
    noise = np.random.randint(1, 32, intensity_image.shape, dtype=np.uint8)
    intensity_image = np.clip(intensity_image + noise, 0, 255).astype(np.uint8)
    timer.log_time()
    return intensity_image


def get_result_range() -> range:
    if GlobalState.get_current_model() in [Model.SHORT_RANGE, Model.IMAGING]:
        return range(0, 1)
    else:
        return range(0, 4)


def stopped_stream() -> None:
    result_queues.flush()
    receive_queues.flush()
    target_queues.flush()
    range_doppler_info.reset()
    while converter_run.is_set() and GlobalState.is_stopped():
        range_doppler_info.reset_frame_rate
        receive_queues.flush()
        target_queues.flush()
        stop_buf: memoryview[int] = STATIC_CONFIG.stopped_buf
        if not result_queues.anyfull():
            for result_idx in get_result_range():
                result_queues[result_idx].put(stop_buf)
        time.sleep(0.04)


def enqueue_range_doppler_result(idx: int, result: NDArray[np.int16], cfar_results: list[DopplerRangeEntry]) -> None:
    if not result_queues[idx].full():
        frame: memoryview[int] = (
            Functor(result).bind(norm_image).bind(heat_map).bind(cfar(cfar_results)).bind(create_frame).value
        )
        result_queues[idx].put(frame)


def enqueue_targets(idx: int, targets: list[AoAEntry]) -> None:
    if not target_queues[idx].full():

        target_queues[idx].put(
            [
                {
                    "x": -1 * t.y,  # pyright: ignore [reportAny]
                    "y": t.x,  # pyright: ignore [reportAny]
                    "z": t.z + 4,  # pyright: ignore [reportAny]
                    "velocity": t.velocity,  # pyright: ignore [reportAny]
                }
                for t in targets
            ]
        )


def hw_stream():
    while converter_run.is_set() and not GlobalState.is_stopped() and GlobalState.use_hw():
        for idx in get_result_range():
            try:
                result: DataBlob = receive_queues[idx].get(timeout=0.06)
                num_cfar_results: int = result.cfar_header.length
                cfar_results: list[DopplerRangeEntry] = result.cfar_payload[:num_cfar_results]
                num_targets: int = result.aoa_header.length
                targets: list[AoAEntry] = result.aoa_payload[:num_targets]
                logger.debug(f"Found {num_cfar_results} CFAR results and {num_targets} targets")
                logger.debug(f"Aoa Header: Minus1 {result.aoa_header.minus1}, Length {result.aoa_header.length}")
                for t in targets:
                    logger.debug(f"Target: X {t.x:.2f}, Y {t.y:.2f}, Z {t.z:.2f}, Velocity {t.velocity:.2f}")
                for c in cfar_results:
                    logger.debug(f"CFAR: Doppler {c.doppler}, Range {c.range}")
                range_doppler = result.range_doppler_data
                range_doppler_np = np.ctypeslib.as_array(range_doppler)  # pyright: ignore [reportUnknownArgumentType]
                enqueue_range_doppler_result(
                    idx,
                    range_doppler_np.reshape((1024, 512)),  # pyright: ignore [reportUnknownArgumentType]
                    cfar_results,  # pyright: ignore [reportUnknownArgumentType]
                )
                enqueue_targets(idx, targets)  # pyright: ignore [reportUnknownArgumentType]
            except queue.Empty:
                continue


def sw_stream():
    while converter_run.is_set() and not GlobalState.is_stopped() and GlobalState.use_sw():
        if not result_queues.anyfull():
            for idx in get_result_range():
                frame = (
                    Functor(synthetic_result(GlobalState.get_current_steps()[idx], idx))
                    .bind(heat_map)
                    .bind(sw_cfar)
                    .bind(create_frame)
                    .value
                )
                result_queues[idx].put(frame)


def converter():
    while converter_run.is_set():
        stopped_stream()
        hw_stream()
        sw_stream()
    receive_queues.flush()
    result_queues.flush()
    target_queues.flush()
    logger.info("Converter Stopped")


def export_results(intensity_image: NDArray[np.int32], current_step: int, channel: int) -> None:
    if STATIC_CONFIG.export_results:
        result_file_name = f"results/result_channel_{channel}_position_{int(current_step):04d}.bin"
        result_file_path = Path(result_file_name)
        if not result_file_path.is_file():
            intensity_image.tofile(result_file_path)
