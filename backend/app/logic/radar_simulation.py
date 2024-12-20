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
from app.logic.model import Model
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

#######################################################################################################################
# Module Global Variables
#
log_level = logging.NOTSET  # NOTSET, DEBUG, INFO, WARNING, ERROR

logger = logging.getLogger(__name__)
logger_stream = logging.StreamHandler()
logger_stream.setLevel(log_level)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
logger_stream.setFormatter(formatter)
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

    def __getitem__(self, idx: int) -> queue.Queue[T]:
        return self.queues[idx]

    def __iter__(self) -> Generator[queue.Queue[T], None, None]:
        for q in self.queues:
            yield q


result_queues = QueueList(num_queues=4, maxsize=2)
send_queues = QueueList(num_queues=4, maxsize=7)
receive_queues = QueueList(num_queues=4, maxsize=7)

stop_producer = threading.Event()
mutex_lock = threading.Lock()
#
#######################################################################################################################


def reset_fifos() -> bool:
    if STATIC_CONFIG.versal_lib:
        err = STATIC_CONFIG.versal_lib.reset_hw()
        return err == 0
    return True


def flush_queues() -> None:
    logger.debug("Entering")
    send_queues.flush()
    receive_queues.flush()
    result_queues.flush()
    _ = reset_fifos()
    logger.debug("Leaving")


def start_threads(idx: int) -> None:
    logger.debug("Entering")
    global producer
    global consumer
    global converter
    if idx == 0:
        logger.debug("Wait for mutex")
        locked = mutex_lock.acquire(timeout=0.1)
        if locked:
            flush_queues()
            logger.debug("Mutex aquired")
            producer = threading.Thread(target=send_radar_scene, name="producer")
            producer.start()
            consumer = threading.Thread(target=receive_radar_result_loop, name="consumer")
            consumer.start()
            converter = threading.Thread(target=create_radar_result, name="converter")
            converter.start()
    logger.debug("Leaving")


def stop_threads(idx: int) -> None:
    logger.debug("Entering")
    stop_producer.set()
    if idx == 0:
        # converter only stops once producer and consumer is stopped so we only need to wait for the converter
        converter.join()

        range_doppler_info.reset()

        logger.debug("Releasing mutex")
        mutex_lock.release()
    logger.debug("Leaving")


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    logger.debug("Entering")
    while GlobalState.get_current_model() not in [
        Model.SHORT_RANGE,
        Model.QUAD_CORNER,
        Model.IMAGING,
    ]:
        time.sleep(0.001)

    stop_producer.clear()
    start_threads(idx)

    counter = 0
    while not GlobalState.leaving_page():
        counter += 1
        try:
            frame = result_queues[idx].get_nowait()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            time.sleep(0.001)
            continue

    logger.debug(f"Stopping threads {GlobalState.get_current_model()}")
    stop_threads(idx)
    if idx == 0:
        GlobalState.set_left_page()
    logger.debug("Leaving")


def send_scene(mode: int, step: int, emulation: int) -> bool:
    err = -1
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.send_scene(mode, step, emulation)
    return err == 0


def current_send_step() -> int:
    current_steps = GlobalState.get_current_steps()
    # The model in the app and the generation of the radar data are 180° phase shifted
    send_step = int(current_steps[3])
    return int(send_step)


def send_radar_scene():
    timer = Timer("send_radar_scene")
    if GlobalState.has_hw():
        mode = {Model.SHORT_RANGE: 0, Model.QUAD_CORNER: 1, Model.IMAGING: 2}
        while not stop_producer.is_set():
            if GlobalState.use_hw() and GlobalState.is_running():
                if not send_queues[0].full():
                    send_step = current_send_step()
                    if send_scene(mode[GlobalState.get_current_model() or Model.SHORT_RANGE], send_step, 0):
                        timer.log_time()
                        send_queues[0].put(send_step)
                else:
                    time.sleep(0.016)  # queue max size * intended frame rate
            else:
                time.sleep(0.1)
    else:
        while not stop_producer.is_set():
            time.sleep(0.1)

    logger.debug("Stopping sender thread")


def receive_radar_result() -> NDArray[np.int16]:
    complex_result = np.empty((4, 1024, 512, 2), np.int16)
    timer = Timer(name="get_radar_result")
    err = 0
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.receive_result(
            complex_result.ctypes,
            4 * 1024 * 512 * 2 * ctypes.sizeof(ctypes.c_int16),
            0,
        )
    timer.log_time()
    return complex_result if err == 0 else np.zeros((4, 1024, 512)).astype(np.int16)


def receive_radar_result_loop():
    timer = Timer(name="receive loop")
    previous_step = -1
    while producer.is_alive():
        if GlobalState.use_hw():
            try:
                send_step = send_queues[0].get_nowait()
                complex_result = receive_radar_result()
                if send_step != previous_step and not receive_queues.anyfull():
                    receive_queues[0].put(complex_result)
                previous_step = send_step
                range_doppler_info.fps = int(1 / timer.duration())
            except queue.Empty:
                time.sleep(0.001)
        else:
            time.sleep(0.1)

    # clean up hw buffers

    for q in send_queues:
        while not q.empty():
            _ = q.get()
            _ = receive_radar_result()

    logger.debug("Stopping receiver thread")


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


def heat_map(intensity_image: NDArray[np.int32]) -> NDArray[np.uint8]:
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

    for rgb in range(0, 3):
        rgb_image[coord[0] - size : coord[0] - mask_size, coord[1] - width : coord[1] + width, rgb] = color[rgb]
        rgb_image[coord[0] + mask_size : coord[0] + size, coord[1] - width : coord[1] + width, rgb] = color[rgb]

        rgb_image[coord[0] - width : coord[0] + width, coord[1] - size : coord[1] - mask_size, rgb] = color[rgb]
        rgb_image[coord[0] - width : coord[0] + width, coord[1] + mask_size : coord[1] + size, rgb] = color[rgb]


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
    for rgb in range(0, 3):
        rgb_image[coord[0] - size, coord[1] - size : coord[1] + size, rgb] = color[rgb]
        rgb_image[coord[0] + size, coord[1] - size : coord[1] + size, rgb] = color[rgb]
        rgb_image[coord[0] - size : coord[0] + size, coord[1] + size, rgb] = color[rgb]
        rgb_image[coord[0] - size : coord[0] + size, coord[1] - size, rgb] = color[rgb]


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


def synthetic_result(current_step: int, channel: int) -> NDArray[np.int32]:
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
    intensity_image = np.clip(intensity_image + noise, 0, 255).astype(np.int32)
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
    while consumer.is_alive() and GlobalState.is_stopped():
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
        results = None
        try:
            results = receive_queues[0].get_nowait()
            if not result_queues.anyfull():
                if results.shape != (4, 1024, 512, 2):
                    logger.error(
                        f"Result shape is not as expected:: Expected: (4, 1024, 512, 2) -> Actual: {results.shape}"
                    )
                    continue
                intensity_images = convert_to_intensity_image(complex_result=results)
                for result_idx in get_result_range():
                    frame = cfar(heat_map(intensity_images[result_idx, ...]))
                    result_queues[result_idx].put(create_frame(frame))
        except queue.Empty:
            time.sleep(0.001)
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
