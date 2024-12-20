# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAny=false
import ctypes
import logging
import queue
import threading
import time
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Any, Generic, TypeVar, final

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.logic.config import STATIC_CONFIG
from app.logic.model import MODEL_LOOKUP, Model
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

#######################################################################################################################
# Module Global Variables
#
log_level = logging.ERROR  # NOTSET, DEBUG, INFO, WARNING, ERROR


class CustomFormatter(logging.Formatter):

    grey: str = "\x1b[38;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = "%(levelname)-8s (%(filename)-26s:%(lineno)3d:%(funcName)-30s) - %(message)s "

    FORMATS: dict[int, str] = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord):  # pyright: ignore [reportImplicitOverride]
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger_stream = logging.StreamHandler()
logger_stream.setLevel(log_level)
# formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
logger_stream.setFormatter(CustomFormatter())
logger.setLevel(log_level)
logger.addHandler(logger_stream)
logger.propagate = False

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
send_queue = queue.Queue()  # no max size as hw should give back pressure
receive_queues = QueueList(num_queues=4, maxsize=2)

stop_producer = threading.Event()
mutex_lock = threading.Lock()

gen_frames_state = [threading.Event(), threading.Event(), threading.Event(), threading.Event()]
#
#######################################################################################################################


def flush_card(timeout_ms: int) -> bool:
    if STATIC_CONFIG.versal_lib:
        err = STATIC_CONFIG.versal_lib.flush(timeout_ms)
        return err == 0
    return True


def flush(q: queue.Queue[T]):
    while not q.empty():
        _ = q.get()


def flush_queues() -> None:
    logger.debug("Entering")
    flush(send_queue)
    receive_queues.flush()
    result_queues.flush()
    _ = flush_card(400)
    logger.debug("Leaving")


def start_threads(idx: int) -> None:
    logger.debug(f"Entering START THREADS with idx = {idx}")
    global producer
    global consumer
    global converter
    if idx == 0:
        logger.debug("Wait for mutex")
        locked = mutex_lock.acquire(timeout=0.1)
        if locked:
            flush_queues()
            logger.debug("-- Mutex aquired --")
            producer = threading.Thread(target=send_radar_scene, name="producer")
            producer.start()
            consumer = threading.Thread(target=receive_radar_result_loop, name="consumer")
            consumer.start()
            converter = threading.Thread(target=create_radar_result, name="converter")
            converter.start()
        else:
            logger.error("!! Mutex NOT aquired!!")
    logger.debug(f"Leaving START THREADS with idx = {idx}")


def stop_threads(idx: int) -> None:
    logger.debug("Entering")
    stop_producer.set()
    logger.info("set stop_producer")
    if idx == 0:
        producer.join()
        consumer.join()
        converter.join()

        range_doppler_info.reset()

        logger.debug("Releasing mutex")
        mutex_lock.release()
    logger.debug("Leaving")


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    logger.debug(f"Entering GEN FRAMES with idx = {idx}")
    gen_frames_state[idx].set()
    GlobalState.set_entered_page()

    stop_producer.clear()
    start_threads(idx)

    while not GlobalState.leaving_page():
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
            for i,s in enumerate(gen_frames_state):
                print(f"state is set of [{i}] = {s.is_set()}")
            time.sleep(0.001)
        GlobalState.set_left_page()
    logger.debug(f"Leaving GEN FRAMES with idx = {idx}")


def send_scene():
    err = -1
    num_channels = 16 if GlobalState.model == Model.IMAGING else 4
    step = GlobalState.get_current_steps()
    uid = MODEL_LOOKUP[GlobalState.model.value]
    for idx in get_result_range():
        logger.debug(f"sending idx = [{idx}] step = {step[idx]}")
        if STATIC_CONFIG.versal_lib:
            err: int = STATIC_CONFIG.versal_lib.send_scene(idx, uid, step[idx], num_channels, 0)
            if err == 0:
                send_queue.put(step[0])
            else:
                logger.error("Error sending to card")


def current_send_step() -> int:
    current_steps = GlobalState.get_current_steps()
    # The model in the app and the generation of the radar data are 180° phase shifted
    send_step = int(current_steps[3])
    return int(send_step)


def send_radar_scene():
    logger.debug("Entering")
    timer = Timer("send_radar_scene")
    if GlobalState.has_hw():
        while not stop_producer.is_set():
            logger.info("looping")
            if GlobalState.use_hw() and GlobalState.is_running():
                send_scene()
                timer.log_time()
            else:
                time.sleep(0.1)
    else:
        while not stop_producer.is_set():
            time.sleep(0.1)

    logger.debug("Leaving")


def receive_radar_result() -> tuple[int, int, NDArray[np.int16]]:
    complex_result = np.empty((1024, 512), np.int16)
    timer = Timer(name="get_radar_result")
    err = 0
    idx = ctypes.c_uint32(0)
    uid = ctypes.c_uint32(0)
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.receive_result(
            complex_result.ctypes,
            ctypes.byref(idx),
            ctypes.byref(uid),
            1024 * 512 * ctypes.sizeof(ctypes.c_int16),
            0,
        )
    timer.log_time()
    return (
        (idx.value, uid.value, complex_result) if err == 0 else (0, uid.value, np.zeros((1024, 512)).astype(np.int16))
    )


def check_bundle_step(radar_idx: int, bundle_step: int | None, step: int) -> int:
    if bundle_step and step != bundle_step:
        logger.error(
            f"Unmatched send step within bundle radar[{radar_idx}]->{step}, bundled step expected is {bundle_step}"
        )
    return step


def check_expected_radar_idx(received: int, expected: int):
    if received != expected:
        logger.error(f"Unmatched index for expected result expected = {expected}, actual = {received}")


def receive_radar_result_loop():
    logger.debug("Entering")
    timer = Timer(name="receive loop")
    previous_step = -1
    INTEGRATION_TIME = 50
    count = 0
    while producer.is_alive():
        if GlobalState.use_hw():
            try:
                # check if any is full and only push to queue if all have room. This ensures that the queues stay in sync
                full = receive_queues.anyfull()
                send_step = -1
                bundle_step: None | int = None
                for i in get_result_range():
                    send_step = send_queue.get_nowait()
                    logger.debug(f"receiving step {send_step} radar_idx = {i}")
                    bundle_step = check_bundle_step(i, bundle_step, send_step)
                    idx, uid, complex_result = receive_radar_result()
                    count += 1
                    if count % INTEGRATION_TIME == 0:
                        range_doppler_info.fps = int(INTEGRATION_TIME / timer.duration())
                    if uid != MODEL_LOOKUP[GlobalState.model.value]:
                        logger.error(f"UID model:{uid} current model is {MODEL_LOOKUP[GlobalState.model.value]}")
                    check_expected_radar_idx(idx, i)
                    if send_step != previous_step and not full:
                        logger.debug(f"pushing to receive_queue[{idx}]")
                        receive_queues[i].put(complex_result)
                previous_step = send_step
                # logger.info("SUCCESSFULL RECEIVED RESULT")
            except queue.Empty:
                time.sleep(0.001)
        else:
            time.sleep(0.1)

    # clean up hw buffers
    # dequeue send_queue
    logger.debug("Cleaning up remaining data")
    while not send_queue.empty():
        _ = send_queue.get()
        _ = receive_radar_result()

    logger.debug("Leaving")


def convert_to_intensity_image(complex_result: NDArray[np.int16]) -> NDArray[np.int32]:
    timer = Timer(name="convert_to_intensity_image")
    inphase = complex_result[..., 0].astype(float)
    quadrature = complex_result[..., 1].astype(float)
    power = np.square(inphase) + np.square(quadrature)
    intensity_images = np.log10(power + 1)
    max: float = np.amax(intensity_images)
    intensity_images = intensity_images / max * 255
    intensity_images = np.roll(np.clip(intensity_images, 0, 255).astype(np.int32), 256, axis=2)
    timer.log_time()
    return intensity_images


def norm_image(image: NDArray[np.int16]) -> NDArray[np.uint8]:
    max: np.int16 = np.amax(image)
    norm_image = image / max * 255
    return np.roll(norm_image.astype(np.uint8), 256, axis=1)


def heat_map(intensity_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    timer = Timer(name="heat_map")
    flipped = np.flip(intensity_image, axis=0)
    img = np.empty((*flipped.shape, 3), dtype=np.uint8)
    img[..., 0] = np.where(flipped < 128, 0, (flipped - 128) * 2)  # red
    img[..., 1] = np.where(flipped < 128, flipped * 2, 255)  # green
    img[..., 2] = np.where(flipped < 128, 255 - flipped * 2, 0)  # blue
    timer.log_time()
    return img


def draw_cross(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    size: int,
    mask_size: int,
    width: int,
):
    if width % 2 == 0:
        width = max(width - 1, 0)

    left = max(0, coord[0] - size)
    left_box = max(0, coord[0] - mask_size)
    right = min(1023, coord[0] + size)
    right_box = min(1023, coord[0] + mask_size)

    top = min(511, coord[1] + size)
    top_box = min(511, coord[1] + mask_size)
    bottom = max(0, coord[1] - size)
    bottom_box = max(0, coord[1] - mask_size)

    width_left = max(0, coord[0] - width)
    width_right = min(1023, coord[0] + width)
    width_bottom = max(0, coord[1] - width)
    width_top = min(511, coord[1] + width)

    for rgb in range(0, 3):
        rgb_image[left:left_box, width_bottom:width_top, rgb] = color[rgb]
        rgb_image[right_box:right, width_bottom:width_top, rgb] = color[rgb]

        rgb_image[width_left:width_right, bottom:bottom_box, rgb] = color[rgb]
        rgb_image[width_left:width_right, top_box:top, rgb] = color[rgb]


def draw_box(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    size: int,
    width: int,
):
    for i in range(0, width):
        _draw_box(rgb_image, color, coord, size + i)


def _draw_box(
    rgb_image: NDArray[np.uint8], color: tuple[np.uint8, np.uint8, np.uint8], coord: tuple[np.intp, ...], size: int
):
    left = max(0, coord[0] - size)
    right = min(1023, coord[0] + size)
    top = min(511, coord[1] + size)
    bottom = max(0, coord[1] - size)
    for rgb in range(0, 3):
        rgb_image[left, bottom:top, rgb] = color[rgb]
        rgb_image[right, bottom:top, rgb] = color[rgb]
        rgb_image[left:right, top, rgb] = color[rgb]
        rgb_image[left:right, bottom, rgb] = color[rgb]


def cfar(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if GlobalState.cfar_enabled():
        coord = np.argmax(rgb_image[..., 0])
        shape_coord: tuple[np.intp, ...] = np.unravel_index(coord, rgb_image[..., 0].shape)
        red = (np.uint8(255), np.uint8(0), np.uint8(0))
        size = 5
        width = 2
        draw_box(rgb_image, red, shape_coord, size, width)
        draw_cross(rgb_image, red, shape_coord, 3 * size, size, width)
    return rgb_image


def create_frame(rgb_array: NDArray[np.uint8]):
    timer = Timer(name="create_frame")
    buf = BytesIO()
    image = Image.fromarray(rgb_array, "RGB")
    image.save(buf, "JPEG")
    frame = buf.getbuffer()
    timer.log_time()
    return frame


def synthetic_result(current_step: int, channel: int) -> NDArray[np.uint8]:
    logger.debug("Entering")
    timer = Timer(name="synthetic_result")
    phase: NDArray[np.float32] = 2 * np.pi * current_step / STATIC_CONFIG.number_of_steps_in_period[channel]
    ypos = int((np.cos(phase + np.pi) * 0.9 + 1) / 2 * 1023)
    xpos = 511 + int(
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


def stopped_stream():
    logger.debug("Entering")
    result_queues.flush()
    receive_queues.flush()
    range_doppler_info.reset()
    count = 0
    while consumer.is_alive() and GlobalState.is_stopped():
        count += 1
        if count % 25 == 0:
            logger.info("in stopped_stream")
        receive_queues.flush()
        stop_buf = STATIC_CONFIG.stopped_buf
        if not result_queues.anyfull():
            for result_idx in get_result_range():
                result_queues[result_idx].put(stop_buf)
        time.sleep(0.04)
    logger.debug("Leaving")


def hw_stream():
    logger.debug("Entering")
    while consumer.is_alive() and not GlobalState.is_stopped() and GlobalState.use_hw():
        result = None
        if GlobalState.model == Model.QUAD_CORNER:
            if receive_queues.anyempty():
                time.sleep(0.001)
                continue
        else:
            if receive_queues[0].empty():
                time.sleep(0.001)
                continue

        for idx in get_result_range():
            result = receive_queues[idx].get()
            if not result_queues.anyfull():
                if result.shape != (1024, 512):
                    logger.error(
                        f"Result shape is not as expected:: Expected: (4, 1024, 512, 2) -> Actual: {result.shape}"
                    )
                    continue
                normed_image = norm_image(result)
                if not result_queues[idx].full():
                    frame = create_frame(cfar(heat_map(normed_image)))
                    result_queues[idx].put(frame)
    logger.debug("Leaving")


def sw_stream():
    logger.debug("Entering")
    while consumer.is_alive() and not GlobalState.is_stopped() and not GlobalState.use_hw():
        if not result_queues.anyfull():
            for idx in get_result_range():
                image = synthetic_result(GlobalState.get_current_steps()[idx], idx)
                rgb_image = heat_map(image)
                rgb_image = cfar(rgb_image)
                frame = create_frame(rgb_image)
                result_queues[idx].put(frame)
    logger.debug("Leaving")


def create_radar_result():
    logger.debug("Entering")
    while consumer.is_alive():
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
