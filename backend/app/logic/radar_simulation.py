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
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.logic.config import STATIC_CONFIG
from app.logic.model import Model
from app.logic.state import GlobalState
from app.logic.status import range_doppler_info
from app.logic.timer import Timer

logger = logging.getLogger(__name__)
logger_stream = logging.StreamHandler()
logger_stream.setLevel(logging.INFO)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
logger_stream.setFormatter(formatter)
logger.setLevel(level=logging.NOTSET)
logger.info("radar demo app")
logger.addHandler(logger_stream)
logger.propagate = False

DEBUG_TIME = False


radar_result_queues = [queue.Queue(), queue.Queue(), queue.Queue(), queue.Queue()]
radar_send_queue = queue.Queue(maxsize=7)
radar_receive_queue = queue.Queue(maxsize=7)
stop_producer = threading.Event()
mutex_lock = threading.Lock()


def reset_fifos() -> bool:
    if STATIC_CONFIG.versal_lib:
        err = STATIC_CONFIG.versal_lib.reset_hw()
        return err == 0
    return True


def start_threads(idx: int) -> None:
    logger.info("Starting Threads")
    global producer
    global consumer
    global converter
    if idx == 0:
        logger.info("Wait for mutex")
        locked = mutex_lock.acquire(timeout=0.1)
        if locked:
            logger.info("Mutex aquired")
            _ = reset_fifos()
            producer = threading.Thread(target=send_radar_scene, name="producer")
            producer.start()
            consumer = threading.Thread(target=receive_radar_result_loop, name="consumer")
            consumer.start()
            converter = threading.Thread(target=create_radar_result, name="converter")
            converter.start()


def stop_threads(idx: int) -> None:
    logger.info("Stopping threads")
    stop_producer.set()
    if idx == 0:
        # converter only stops once producer and consumer is stopped so we only need to wait for the converter
        converter.join()

        # Empty queues to have a clean slate
        while not radar_receive_queue.empty():
            _ = radar_receive_queue.get()

        for idx in range(0, 4):
            while not radar_result_queues[idx].empty():
                _ = radar_result_queues[idx].get()

        range_doppler_info.reset()

        logger.info("Releasing mutex")
        mutex_lock.release()


def gen_frames(idx: int) -> Generator[Any, Any, None]:  # pyright: ignore [reportExplicitAny]
    while GlobalState.get_current_model() not in [
        Model.SHORT_RANGE.value,
        Model.QUAD_CORNER.value,
        Model.IMAGING.value,
    ]:
        time.sleep(0.001)

    stop_producer.clear()
    start_threads(idx)

    counter = 0
    while GlobalState.get_current_model() in [Model.SHORT_RANGE.value, Model.QUAD_CORNER.value, Model.IMAGING.value]:
        counter += 1
        try:
            frame = radar_result_queues[idx].get_nowait()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            time.sleep(0.001)
            continue

    logger.info(f"Stopping threads {GlobalState.get_current_model()}")
    stop_threads(idx)


def send_scene(mode: int, step: int, emulation: int) -> bool:
    err = -1
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.send_scene(mode, step, emulation)
    return err == 0


def current_send_step() -> int:
    current_steps = GlobalState.get_current_steps()
    # The model in the app and the generation of the radar data are 180° phase shifted
    send_step = int(
        (current_steps[3] + radar_send_queue.qsize() + radar_receive_queue.qsize() + radar_result_queues[0].qsize())
        % STATIC_CONFIG.number_of_steps_in_period[3]
    )
    return int(send_step)


def send_radar_scene():
    timer = Timer("send_radar_scene")
    if GlobalState.has_hw():
        mode = {Model.SHORT_RANGE.value: 0, Model.QUAD_CORNER.value: 1, Model.IMAGING.value: 2}
        while not stop_producer.is_set():
            if GlobalState.use_hw() and GlobalState.is_running():
                if not radar_send_queue.full():
                    send_step = current_send_step()
                    if send_scene(mode[GlobalState.get_current_model() or Model.SHORT_RANGE.value], send_step, 0):
                        timer.log_time()
                        radar_send_queue.put(send_step)
                else:
                    time.sleep(0.016)  # queue max size * intended frame rate
            else:
                time.sleep(0.1)
    else:
        while not stop_producer.is_set():
            time.sleep(0.1)

    logger.info("Stopping sender thread")


def receive_radar_result(drop: bool) -> None:
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
    if drop:
        return
    else:
        if err != 0:
            radar_receive_queue.put(np.zeros((4, 1024, 512)).astype(np.int16))
        else:
            radar_receive_queue.put(complex_result)


def receive_radar_result_loop():
    timer = Timer(name="receive loop")
    previous_step = -1
    while producer.is_alive():
        if GlobalState.use_hw():
            try:
                send_step = radar_send_queue.get_nowait()
                receive_radar_result(send_step == previous_step)
                previous_step = send_step
                duration = timer.duration()
                range_doppler_info.fps = int(1 / duration)
            except queue.Empty:
                time.sleep(0.001)
                pass
        else:
            time.sleep(0.1)

    # clean up hw buffers
    while not radar_send_queue.empty():
        _ = radar_send_queue.get()
        _ = receive_radar_result(True)

    logger.info("Stopping receiver thread")


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


def create_frame(rgb_array: NDArray[np.uint8]):
    timer = Timer(name="create_frame")
    buf = BytesIO()
    image = Image.fromarray(rgb_array, "RGB")
    image.save(buf, "JPEG")
    frame = buf.getbuffer()
    timer.log_time()
    return frame


def synthetic_result(current_step: int, channel: int) -> NDArray[np.int32]:
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
    return intensity_image


def create_radar_result():
    while consumer.is_alive():
        if GlobalState.is_stopped():
            while not radar_receive_queue.empty():
                _ = radar_receive_queue.get()
            range_doppler_info.reset()
            stop_buf = STATIC_CONFIG.stopped_buf
            if GlobalState.get_current_model() == Model.SHORT_RANGE.value:
                radar_result_queues[0].put(stop_buf)
            else:
                for q in radar_result_queues:
                    q.put(stop_buf)
            time.sleep(0.04)

        elif GlobalState.use_hw():
            results = None
            for _ in range(radar_receive_queue.qsize() // 2):
                _ = radar_receive_queue.get()
            if not radar_receive_queue.empty():
                results = radar_receive_queue.get()
            if results is not None:
                if results.shape != (4, 1024, 512, 2):
                    continue
                intensity_images = convert_to_intensity_image(complex_result=results)
                frames: list[NDArray[np.uint8]] = []
                for index in range(0, 4):
                    frames.append(heat_map(intensity_images[index, ...]))

                if GlobalState.get_current_model() == Model.SHORT_RANGE.value:
                    radar_result_queues[0].put(create_frame(frames[0]))
                else:
                    for idx, q in enumerate(radar_result_queues):
                        q.put(create_frame(frames[idx]))
            else:
                time.sleep(0.001)

        elif not GlobalState.use_hw():
            if GlobalState.get_current_model() == Model.SHORT_RANGE.value:
                result_range = range(0, 1)
            else:
                result_range = range(0, 4)

            for idx in result_range:
                image = synthetic_result(GlobalState.get_current_steps()[idx], idx)
                rgb_image = heat_map(image)
                frame = create_frame(rgb_image)
                radar_result_queues[idx].put(frame)


def export_results(intensity_image: NDArray[np.int32], current_step: int, channel: int) -> None:
    if STATIC_CONFIG.export_results:
        result_file_name = f"results/result_channel_{channel}_position_{int(current_step):04d}.bin"
        result_file_path = Path(result_file_name)
        if not result_file_path.is_file():
            logger.info(f"writing to file {result_file_name}")
            intensity_image.tofile(result_file_path)
