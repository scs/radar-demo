# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAny=false
from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, final

import numpy as np
from numpy.typing import NDArray

from app.logic.cfar import cfar
from app.logic.config import STATIC_CONFIG
from app.logic.flush_card import eib, eob, oib, oob
from app.logic.image_utils import create_frame, heat_map, norm_image
from app.logic.logging import LogLevel, get_logger
from app.logic.model import Model
from app.logic.output_exception import InputFull, OutputEmpty
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

logger = get_logger(__name__, LogLevel.WARNING)
T = TypeVar("T")
U = TypeVar("U")


class Functor(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value: T = value

    def bind(self, func: Callable[[T], U]) -> Functor[U]:
        return Functor(func(self.value))


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

producer_run = threading.Event()
receiver_run = threading.Event()
converter_run = threading.Event()

# locks to make sure only one of the multiple gen_frames can start / stop the threads
stop_lock: threading.Lock = threading.Lock()
start_lock: threading.Lock = threading.Lock()

gen_frames_state = [threading.Event(), threading.Event(), threading.Event(), threading.Event()]


#
#######################################################################################################################


def flush_queues() -> None:
    logger.debug("Entering")
    receive_queues.flush()
    result_queues.flush()
    # _ = flush_card(400)
    logger.debug("Leaving")


def start_threads(idx: int) -> None:
    logger.debug(f"Entering START THREADS with idx = {idx}")
    global sender_thread
    global receiver_thread
    global converter_thread
    logger.debug("Wait for mutex")
    if start_lock.acquire(blocking=False):
        producer_run.set()
        receiver_run.set()
        converter_run.set()
        flush_queues()
        logger.debug(f"-- Mutex aquired for thread with id = {idx} --")
        sender_thread = threading.Thread(target=sender, name="producer")
        sender_thread.start()
        receiver_thread = threading.Thread(target=receiver, name="consumer")
        receiver_thread.start()
        converter_thread = threading.Thread(target=converter, name="converter")
        converter_thread.start()
    logger.debug(f"Leaving START THREADS with idx = {idx}")


def stop_threads(idx: int) -> None:
    logger.debug(f"Entering STOP THREADS with idx = {idx}")
    _ = stop_lock.acquire()
    if start_lock.locked():
        producer_run.clear()
        logger.info("set stop_producer")
        sender_thread.join()
        receiver_thread.join()
        converter_thread.join()

        range_doppler_info.reset()

        logger.debug("Releasing mutex")
        start_lock.release()
    stop_lock.release()
    logger.debug(f"Leaving STOP THREADS with idx = {idx}")


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    logger.debug(f"Entering GEN FRAMES with idx = {idx}")
    gen_frames_state[idx].set()

    start_threads(idx)

    while not GlobalState.stop_producer.is_set():
        try:
            frame = result_queues[idx].get_nowait()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            time.sleep(0.001)
            continue

    logger.debug(f"[{idx}] Stopping threads {GlobalState.get_current_model()}")
    stop_threads(idx)
    gen_frames_state[idx].clear()
    if idx == 0:
        while any([x.is_set() for x in gen_frames_state]):
            for i, s in enumerate(gen_frames_state):
                if s.is_set():
                    logger.warning(f"state is set of [{i}] = {s.is_set()}")
                time.sleep(0.001)
        GlobalState.stop_producer.clear()
    logger.debug(f"Leaving GEN FRAMES with idx = {idx}")


def send_scene(timeout_ms: float, frame_nr: int) -> int:
    num_channels = 16 if GlobalState.model == Model.IMAGING else 4
    step: list[int] = GlobalState.get_current_steps()
    timeout = Timer("send_timeout")
    for idx in get_result_range():
        timeout.start()  # each radar has the same timeout
        logger.debug(f"sending idx = [{idx}] step = {step[idx]}")
        if STATIC_CONFIG.versal_lib:
            while not STATIC_CONFIG.versal_lib.input_ready():
                if timeout.snapshot() / 1000 > timeout_ms:
                    logger.error("Input full timeout")
                    raise InputFull()
                else:
                    time.sleep(0.01)

            err = 500
            for _ in range(3):  # Try max three times to send the same data
                err = STATIC_CONFIG.versal_lib.send_scene(idx, frame_nr, step[idx], num_channels, 0)
                if err == 0:
                    break

            if err:
                logger.error("####################################################################")
                logger.error("####################    Unable to send scene    ####################")
                logger.error("####################################################################")
            else:
                frame_nr += 1

    return frame_nr


def current_send_step() -> int:
    current_steps = GlobalState.get_current_steps()
    send_step = int(current_steps[3])
    return int(send_step)


def sender():
    logger.debug("Entering")
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
            eib(LogLevel.INFO)
            eob(LogLevel.INFO)
            oib(LogLevel.INFO)
            oob(LogLevel.INFO)
            time.sleep(0.1)

    receiver_run.clear()
    logger.debug("Leaving")


def receive_radar_result() -> tuple[int, int, int, NDArray[np.int16]]:
    complex_result = np.empty((1024, 512), np.int16)
    timer = Timer(name="get_radar_result")
    err = 0
    idx = ctypes.c_uint32(0)
    step = ctypes.c_uint32(0)
    frame_nr = ctypes.c_uint32(0)
    if STATIC_CONFIG.versal_lib:
        if STATIC_CONFIG.versal_lib.output_ready():
            err: int = STATIC_CONFIG.versal_lib.receive_result(
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
    return (
        (idx.value, step.value, frame_nr.value, complex_result)
        if err == 0
        else (0, step.value, frame_nr.value, np.zeros((1024, 512)).astype(np.int16))
    )


def make_update() -> Callable[[Timer], None]:
    count = 0

    def update_status(timer: Timer) -> None:
        nonlocal count
        INTEGRATION_TIME = 2 * 4
        if count % INTEGRATION_TIME == 0:
            range_doppler_info.fps = int(INTEGRATION_TIME / timer.duration() / get_result_range().stop)
        count += 1

    return update_status


def make_enqueue() -> Callable[[int, int, NDArray[np.int16]], None]:
    previous_step = -1
    commit = True

    def enqueue(radar_idx: int, step: int, data: NDArray[np.int16]) -> None:
        nonlocal commit
        nonlocal previous_step
        logger.debug(f"Enqueu for idx = {radar_idx}")
        # pre condition
        if radar_idx == 0:
            commit = not receive_queues.anyfull() and step != previous_step
            logger.debug(f"Commit is being set to {commit}, send_step = {step}")
            previous_step = step

        if commit:
            logger.debug(f"Commiting for queue {radar_idx}")
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


def flush_output_buffers() -> None:
    start = time.time()

    if STATIC_CONFIG.versal_lib:
        eib: int = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
        eob: int = STATIC_CONFIG.versal_lib.num_empty_output_buffers()
        logger.info(f"[{eib}] empty input buffers\n[{eob}] empty output buffers")
        while eib != 8 and eob != 8:
            try:
                _, _, _, _ = receive_radar_result()
                eib = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
                eob = STATIC_CONFIG.versal_lib.num_empty_output_buffers()
                stop = time.time()
                logger.info(f"[{eib}] empty buffers\n[{eob}] empty buffers")
                if stop - start > 2:
                    start = time.time()
                    logger.error(f"[{eib}] empty buffers\n[{eob}] empty buffers")
            except OutputEmpty:
                eib = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
                eob = STATIC_CONFIG.versal_lib.num_empty_output_buffers()
                stop = time.time()
                logger.info(f"[{eib}] empty buffers\n[{eob}] empty buffers")
                if stop - start > 2:
                    start = time.time()
                    logger.error(f"[{eib}] empty buffers\n[{eob}] empty buffers")
                continue


def receiver() -> None:
    logger.debug("Entering")
    timer = Timer(name="receive loop")
    check_frame_nr = make_check(lambda x: x + 1)
    check_radar_idx = make_check(lambda x: (x + 1) % get_result_range().stop)
    enqueue_received = make_enqueue()
    update_status = make_update()
    while receiver_run.is_set():
        if GlobalState.has_hw():
            try:
                radar_idx, step, frame_nr, data = receive_radar_result()
                logger.debug(f"received idx {radar_idx}, step {step}, frame_nr {frame_nr}")
                check_radar_idx(radar_idx)
                check_frame_nr(frame_nr)
                enqueue_received(radar_idx, step, data)
                update_status(timer)
            except OutputEmpty:
                time.sleep(0.008)  # wait for 8 ms (half the time that one cycle should take)
        else:
            time.sleep(0.1)

    flush_output_buffers()
    converter_run.clear()
    logger.debug("Leaving")


def synthetic_result(current_step: int, channel: int) -> NDArray[np.uint8]:
    logger.debug("Entering")
    timer: Timer = Timer(name="synthetic_result")
    phase: NDArray[np.float32] = 2 * np.pi * current_step / STATIC_CONFIG.number_of_steps_in_period[channel]
    ypos: int = int((np.cos(phase + np.pi) * 0.9 + 1) / 2 * 1023)
    xpos: int = 511 + int(
        (np.sin(phase)) * 480 / (STATIC_CONFIG.period_in_seconds[channel] / (STATIC_CONFIG.period_in_seconds[0] - 0.5))
    )

    Y, X = np.ogrid[: STATIC_CONFIG.video_dim, : STATIC_CONFIG.video_dim]
    dist_from_center = np.sqrt(np.square(X - xpos) + np.square(Y - ypos))
    norm_dist_from_center = dist_from_center / STATIC_CONFIG.video_dim

    norm_intensity_image = np.power(1 - norm_dist_from_center, 24)
    intensity_image = norm_intensity_image * 255
    noise = np.random.randint(1, 32, intensity_image.shape, dtype=np.uint8)
    intensity_image = np.clip(intensity_image + noise, 0, 255).astype(np.uint8)
    timer.log_time()
    logger.debug("Leaving")
    return intensity_image


def get_result_range() -> range:
    if GlobalState.get_current_model() in [Model.SHORT_RANGE, Model.IMAGING]:
        return range(0, 1)
    else:
        return range(0, 4)


def stopped_stream() -> None:
    logger.debug("Entering")
    result_queues.flush()
    receive_queues.flush()
    range_doppler_info.reset()
    count = 0
    while converter_run.is_set() and GlobalState.is_stopped():
        count += 1
        if count % 25 == 0:
            logger.debug("in stopped_stream")
        range_doppler_info.reset_frame_rate
        receive_queues.flush()
        stop_buf: memoryview[int] = STATIC_CONFIG.stopped_buf
        if not result_queues.anyfull():
            for result_idx in get_result_range():
                result_queues[result_idx].put(stop_buf)
        time.sleep(0.04)
    logger.debug("Leaving")


def shape_ok(result: NDArray[np.int16]) -> bool:
    if result.shape != (1024, 512):
        logger.error(f"Result shape is not as expected:: Expected: (4, 1024, 512, 2) -> Actual: {result.shape}")
        return False
    return True


def enqueue_result(idx: int, result: NDArray[np.int16]) -> None:
    if not result_queues[idx].full() and shape_ok(result):
        frame: memoryview[int] = Functor(result).bind(norm_image).bind(heat_map).bind(cfar).bind(create_frame).value
        logger.info(f"Push to result queue{idx}")
        result_queues[idx].put(frame)


def hw_stream():
    logger.info("Entering")
    while converter_run.is_set() and not GlobalState.is_stopped() and GlobalState.use_hw():
        for idx in get_result_range():
            logger.info(f"Fetching result from recieve queue {idx}")
            try:
                result: NDArray[np.int16] = receive_queues[idx].get(timeout=0.06)
                enqueue_result(idx, result)
            except queue.Empty:
                continue

    logger.info("Leaving")


def sw_stream():
    logger.debug("Entering")
    while converter_run.is_set() and not GlobalState.is_stopped() and GlobalState.use_sw():
        if not result_queues.anyfull():
            for idx in get_result_range():
                frame = (
                    Functor(synthetic_result(GlobalState.get_current_steps()[idx], idx))
                    .bind(heat_map)
                    .bind(cfar)
                    .bind(create_frame)
                    .value
                )
                result_queues[idx].put(frame)
    logger.debug("Leaving")


def converter():
    logger.debug("Entering")
    while converter_run.is_set():
        stopped_stream()
        hw_stream()
        sw_stream()
    receive_queues.flush()
    result_queues.flush()
    logger.debug("Leaving")


def export_results(intensity_image: NDArray[np.int32], current_step: int, channel: int) -> None:
    if STATIC_CONFIG.export_results:
        result_file_name = f"results/result_channel_{channel}_position_{int(current_step):04d}.bin"
        result_file_path = Path(result_file_name)
        if not result_file_path.is_file():
            logger.debug(f"writing to file {result_file_name}")
            intensity_image.tofile(result_file_path)
