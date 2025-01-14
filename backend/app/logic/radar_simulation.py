# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportAny=false
import ctypes
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
from app.logic.flush_card import flush_card
from app.logic.logging import LogLevel, get_logger
from app.logic.model import Model
from app.logic.output_exception import InputFull, OutputEmpty
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

logger = get_logger(__name__, LogLevel.WARNING)
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

stop_producer = threading.Event()

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
    _ = flush_card(400)
    logger.debug("Leaving")


def start_threads(idx: int) -> None:
    logger.debug(f"Entering START THREADS with idx = {idx}")
    global producer
    global consumer
    global converter
    logger.debug("Wait for mutex")
    if start_lock.acquire(blocking=False):
        flush_queues()
        logger.debug(f"-- Mutex aquired for thread with id = {idx} --")
        producer = threading.Thread(target=send_radar_scene, name="producer")
        producer.start()
        consumer = threading.Thread(target=receive_radar_result_loop, name="consumer")
        consumer.start()
        converter = threading.Thread(target=create_radar_result, name="converter")
        converter.start()
    logger.debug(f"Leaving START THREADS with idx = {idx}")


def stop_threads(idx: int) -> None:
    logger.debug(f"Entering STOP THREADS with idx = {idx}")
    _ = stop_lock.acquire()
    if start_lock.locked():
        stop_producer.set()
        logger.info("set stop_producer")
        producer.join()
        consumer.join()
        converter.join()

        range_doppler_info.reset()

        logger.debug("Releasing mutex")
        start_lock.release()
    stop_lock.release()
    logger.debug(f"Leaving STOP THREADS with idx = {idx}")


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    logger.debug(f"Entering GEN FRAMES with idx = {idx}")
    gen_frames_state[idx].set()

    stop_producer.clear()
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
                logger.warning(f"state is set of [{i}] = {s.is_set()}")
                time.sleep(0.001)
        GlobalState.stop_producer.clear()
    logger.debug(f"Leaving GEN FRAMES with idx = {idx}")


def send_scene(idx: int, step: int, ref_step: int) -> int:
    err: int = -1
    num_channels = 16 if GlobalState.model == Model.IMAGING else 4
    logger.debug(f"sending idx = [{idx}] step = {step}")
    if STATIC_CONFIG.versal_lib:
        if STATIC_CONFIG.versal_lib.input_ready():
            err = STATIC_CONFIG.versal_lib.send_scene(idx, ref_step, step, num_channels, 0)
        else:
            # logger.warning("No empty input buffer available")
            raise InputFull()
    return err


def current_send_step() -> int:
    current_steps = GlobalState.get_current_steps()
    # The model in the app and the generation of the radar data are 180° phase shifted
    send_step = int(current_steps[3])
    return int(send_step)


def send_radar_scene():
    logger.debug("Entering")
    timer: Timer = Timer("send_radar_scene")
    range_stop: int = get_result_range().stop
    idx: int = 0
    step: list[int] = GlobalState.get_current_steps()
    if GlobalState.has_hw():
        while not stop_producer.is_set():
            if GlobalState.use_hw() and GlobalState.is_running():
                try:
                    if idx == 0:
                        step = GlobalState.get_current_steps()
                    err = send_scene(idx, step=step[idx], ref_step=step[0])
                    idx = (idx + 1) % range_stop
                    if err != 0:
                        logger.error("Error sending to card")
                    else:
                        timer.log_time()
                except InputFull:
                    time.sleep(0.01)
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
        if STATIC_CONFIG.versal_lib.output_ready():
            err: int = STATIC_CONFIG.versal_lib.receive_result(
                complex_result.ctypes,
                ctypes.byref(idx),
                ctypes.byref(uid),
                1024 * 512 * ctypes.sizeof(ctypes.c_int16),
                0,
            )
        else:
            # logger.warning("No occupied output buffer available")
            raise OutputEmpty()
    timer.log_time()
    return (
        (idx.value, uid.value, complex_result) if err == 0 else (0, uid.value, np.zeros((1024, 512)).astype(np.int16))
    )


def check_received_meta_data(iteration_idx: int, radar_idx: int, step: int, bundle_step: int) -> int:
    check_expected_radar_idx(radar_idx, iteration_idx)
    return check_bundle_step(radar_idx, step, bundle_step)


def check_expected_radar_idx(received: int, iteration_idx: int) -> None:
    if received != iteration_idx % get_result_range().stop:
        logger.error(f"Unmatched index for expected result expected = {iteration_idx}, actual = {received}")


def check_bundle_step(radar_idx: int, send_step: int, bundle_step: int) -> int:
    if radar_idx == 0:
        return send_step
    else:
        if send_step != bundle_step:
            logger.error(
                f"Unmatched send step within bundle radar[{radar_idx}]->{send_step}, bundled step expected is {bundle_step}"
            )
        return bundle_step


def update_status(timer: Timer, count: int) -> None:
    INTEGRATION_TIME = 2 * 4
    if count % INTEGRATION_TIME == 0:
        range_doppler_info.fps = int(INTEGRATION_TIME / timer.duration() / get_result_range().stop)


def enqueue_received(
    radar_idx: int, send_step: int, previous_step: int, full: bool, data: NDArray[np.int16]
) -> tuple[bool, int]:
    # pre condition
    if radar_idx == 0:
        full = receive_queues.anyfull()

    if not full:
        if send_step != previous_step:
            receive_queues[radar_idx].put(data)

    # post condition
    if radar_idx == get_result_range().stop - 1:
        previous_step = send_step

    return full, previous_step


def receive_radar_result_loop() -> None:
    logger.debug("Entering")
    timer = Timer(name="receive loop")
    previous_step: int = -1
    bundle_step: int = 0
    iteration_idx = 0
    full = True
    while producer.is_alive():
        if GlobalState.use_hw():
            try:
                radar_idx, send_step, result = receive_radar_result()
                bundle_step = check_received_meta_data(iteration_idx, radar_idx, send_step, bundle_step)

                full, previous_step = enqueue_received(radar_idx, send_step, previous_step, full, result)
                update_status(timer, iteration_idx)
                iteration_idx += 1
            except OutputEmpty:
                time.sleep(0.01)
        else:
            time.sleep(0.1)

    if STATIC_CONFIG.versal_lib:
        eib: int = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
        eob: int = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
        while eib != 8 and eob != 8:
            try:
                _, _, _ = receive_radar_result()
                eib = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
                eob = STATIC_CONFIG.versal_lib.num_empty_input_buffers()
            except OutputEmpty:
                continue

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
    height: int,
    width: int,
    mask_height: int,
    mask_width: int,
    weight: int,
):
    if weight % 2 == 0:
        weight = max(weight - 1, 0)

    bottom: int = int(max(0, coord[0] - height))
    bottom_box: int = int(max(0, coord[0] - mask_height))
    top: int = int(min(1023, coord[0] + height))
    top_box: int = int(min(1023, coord[0] + mask_height))

    left: int = int(min(511, coord[1] + width))
    left_box: int = int(min(511, coord[1] + mask_width))
    right: int = int(max(0, coord[1] - width))
    right_box: int = int(max(0, coord[1] - mask_width))

    width_bottom_: int = int(max(0, coord[0] - weight))
    width_top_: int = int(min(1023, coord[0] + weight))
    width_left: int = int(max(0, coord[1] - weight))
    width_right: int = int(min(511, coord[1] + weight))

    for rgb in range(0, 3):
        rgb_image[bottom:bottom_box, width_left:width_right, rgb] = color[rgb]
        rgb_image[top_box:top, width_left:width_right, rgb] = color[rgb]

        rgb_image[width_bottom_:width_top_, right:right_box, rgb] = color[rgb]
        rgb_image[width_bottom_:width_top_, left_box:left, rgb] = color[rgb]


def draw_box(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    height: int,
    width: int,
    weight: int,
):
    for i in range(0, weight):
        _draw_box(rgb_image, color, coord, height + i, width + i)


def _draw_box(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    height: int,
    width: int,
):
    bottom: int = int(max(0, coord[0] - height))
    top: int = int(min(1023, coord[0] + height))
    right: int = int(min(511, coord[1] + width))
    left: int = int(max(0, coord[1] - width))
    for rgb in range(0, 3):
        rgb_image[bottom, left:right, rgb] = color[rgb]
        rgb_image[top, left:right, rgb] = color[rgb]
        rgb_image[bottom:top, right, rgb] = color[rgb]
        rgb_image[bottom:top, left, rgb] = color[rgb]


def cfar(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if GlobalState.cfar_enabled():
        coord = np.argmax(rgb_image[..., 0])
        shape_coord: tuple[np.intp, ...] = np.unravel_index(coord, rgb_image[..., 0].shape)
        red: tuple[np.uint8, np.uint8, np.uint8] = (np.uint8(255), np.uint8(0), np.uint8(0))
        if GlobalState.model == Model.QUAD_CORNER:
            width = 5
            height = 10
            weight = 1
        else:
            width = 3
            height = 6
            weight = 2
        draw_box(rgb_image, red, shape_coord, height, width, weight)
        draw_cross(rgb_image, red, shape_coord, 3 * height, 3 * width, height, width, weight)
    return rgb_image


def create_frame(rgb_array: NDArray[np.uint8]):
    timer: Timer = Timer(name="create_frame")
    buf: BytesIO = BytesIO()
    image: Image.Image = Image.fromarray(rgb_array, "RGB")
    image.save(buf, "JPEG")
    frame: memoryview[int] = buf.getbuffer()
    timer.log_time()
    return frame


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
    while consumer.is_alive() and GlobalState.is_stopped():
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


def hw_stream():
    logger.debug("Entering")
    while consumer.is_alive() and not GlobalState.is_stopped() and GlobalState.use_hw():
        if GlobalState.model == Model.QUAD_CORNER:
            if receive_queues.anyempty():
                time.sleep(0.001)
                continue
        else:
            if receive_queues[0].empty():
                time.sleep(0.001)
                continue

        for idx in get_result_range():
            result: NDArray[np.int16] = receive_queues[idx].get()
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
