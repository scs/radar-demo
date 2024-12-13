# pyright: reportAny=false
import ctypes
import logging
import queue
import threading
import time
from collections.abc import Generator
from io import BytesIO
from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from app.logic.config import STATIC_CONFIG
from app.logic.model import Model
from app.logic.state import GlobalState
from app.logic.status import benchmark_info
from app.logic.timer import Timer

matplotlib.use("agg")


def get_current_phase(samples: int) -> NDArray[np.float32]:
    logger.debug("Entering")
    current_step = GlobalState.get_current_steps()[0]
    border = 20
    position: int = abs((STATIC_CONFIG.number_of_steps_in_period[0] / 2) - current_step)
    offseted_position: int = (
        STATIC_CONFIG.number_of_steps_in_period[0]
        + border
        - (STATIC_CONFIG.number_of_steps_in_period[0] / 2 - position)
    )
    norm_value: int = STATIC_CONFIG.number_of_steps_in_period[0] + border
    phase = np.linspace(0 + border, samples, samples) * (2 * np.pi * (offseted_position / norm_value) * 2)
    logger.debug("Leaving")
    return phase


def get_current_signal(samples: int, amplitude: int) -> NDArray[np.int32]:
    logger.debug("Entering")
    phase = get_current_phase(samples)
    rand_noise = np.random.normal(scale=0.15, size=samples)
    signal = np.exp(1.0j * (phase + rand_noise)) * amplitude
    logger.debug("Leaving")
    return signal


def setup_plot(samples: int, amplitude: int) -> tuple[Figure, Line2D, NDArray[np.int32], float]:
    logger.debug("Entering")
    x = np.arange(0, samples)
    signal = get_current_signal(samples, amplitude)
    fft_data = np.fft.fft(signal, samples, norm="forward")
    fft_data = fft_data * samples
    _ = plt.ioff()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    absval = np.abs(fft_data)
    [line] = ax.plot(x, absval)
    line.set_data(x, absval)
    _ = ax.axis("off")
    logger.debug("Leaving")
    return (fig, line, x, np.max(absval))


#######################################################################################################################
# Module Global Variables
#

log_level = logging.NOTSET  # NOTSET, DEBUG, INFO, WARNING, ERROR

logger = logging.getLogger(__name__)
logger_stream = logging.StreamHandler()
logger_stream.setLevel(log_level)
formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
logger_stream.setFormatter(formatter)
logger.setLevel(level=log_level)
logger.info("benchmark demo app")
logger.addHandler(logger_stream)
logger.propagate = False

result_queue = queue.Queue(maxsize=7)
send_queue = queue.Queue(maxsize=7)
receive_queue = queue.Queue(maxsize=7)
stop_producer = threading.Event()
mutex_lock = threading.Lock()

AMPLITUDE = 32
SAMPLES = 512
SCALE = SAMPLES * AMPLITUDE
count = 0
fig, line, x, maxval = setup_plot(SAMPLES, AMPLITUDE)
#
#######################################################################################################################


def send_1d_fft_data(data: NDArray[np.int16], data_size_in_bytes: int, batch_size: int) -> int:
    logger.debug("Entering")
    err = 1
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.send_1d_fft_data(
            data.ctypes,
            data_size_in_bytes,
            batch_size,
        )
    logger.debug("Leaving")
    return err


def receive_1d_fft_results(data: NDArray[np.int16], size: int) -> None:
    logger.debug("Entering")
    if STATIC_CONFIG.versal_lib:
        STATIC_CONFIG.versal_lib.receive_1d_fft_results(data.ctypes, size)
    logger.debug("Leaving")


def reset_fifos() -> bool:
    logger.debug("Entering")
    if STATIC_CONFIG.versal_lib:
        err = STATIC_CONFIG.versal_lib.reset_hw()
        return err == 0
    logger.debug("Leaving")
    return True


def send_data():
    logger.debug("Entering")
    if GlobalState.has_hw():
        while not stop_producer.is_set():
            if GlobalState.use_hw() and GlobalState.is_running():
                if not send_queue.full():
                    signal = get_current_signal(SAMPLES, AMPLITUDE)
                    data_arangement_timer = Timer(name="Data Arangement")
                    batch_size = GlobalState.get_current_batch_size()
                    hw_data = np.zeros((SAMPLES, 2))
                    hw_data[..., 0] = signal.real
                    hw_data[..., 1] = signal.imag
                    hw_data = hw_data.astype(np.int16)
                    data_arangement_timer.log_time()
                    send_timer = Timer(name="PCIe Send")
                    err = send_1d_fft_data(
                        hw_data,
                        2 * SAMPLES * ctypes.sizeof(ctypes.c_int16),
                        batch_size,
                    )
                    if err == 0:
                        send_queue.put(1)
                    send_timer.log_time()
                else:
                    time.sleep(0.004)
            else:
                time.sleep(0.1)
    else:
        while not stop_producer.is_set():
            time.sleep(0.1)
    logger.debug("Leaving")


def receive_result():
    logger.debug("Entering")
    fft_data = np.zeros((SAMPLES, 2)).astype(np.int16)
    receive_1d_fft_results(fft_data, 2 * SAMPLES * ctypes.sizeof(ctypes.c_int16))
    receive_queue.put(fft_data)
    logger.debug("Leaving")


def receive_data():
    logger.debug("Entering")
    global count
    while sender.is_alive():
        if GlobalState.use_hw():
            try:
                _ = send_queue.get_nowait()
                receive_result()
                count = count + 1
            except queue.Empty:
                time.sleep(0.001)
                pass
        else:
            time.sleep(0.1)
    logger.debug("Leaving")


def create_frame(fft_data: NDArray[np.float32]):
    fft_data *= maxval
    line.set_data(x, np.abs(fft_data))

    buf = BytesIO()
    fig.savefig(buf, format="JPEG")

    frame = buf.getbuffer()
    return frame


def hw_stream():
    while receiver.is_alive() and not GlobalState.is_stopped() and GlobalState.use_hw():
        fft_data = None
        while not receive_queue.empty():
            fft_data = receive_queue.get()
        if fft_data is not None:
            plot_timer = Timer(name="Plot")
            real = fft_data[..., 0].astype(float)
            imag = fft_data[..., 1].astype(float)
            amp = np.sqrt(real**2 + imag**2)  # pyright: ignore [reportUnknownArgumentType]
            amp = amp / np.max(amp)
            plot_timer.log_time()
            result_queue.put(create_frame(amp))


def sw_stream():
    while receiver.is_alive() and not GlobalState.is_stopped() and not GlobalState.use_hw():
        signal_timer = Timer(name="signal generation")
        signal = get_current_signal(SAMPLES, AMPLITUDE)
        signal_timer.log_time()
        fft_timer = Timer("NP fft timer")
        fft_data = np.fft.fft(signal, SAMPLES, norm="forward")
        benchmark_info.set_ffts_emulation(int(1 / fft_timer.duration()))
        fft_data = np.abs(fft_data) * SAMPLES
        fft_data = fft_data / np.max(fft_data)
        result_queue.put(create_frame(fft_data))


def stopped_stream():
    while receiver.is_alive() and GlobalState.is_stopped():
        flush_queues()
        benchmark_info.reset()
        frame = STATIC_CONFIG.stopped_buf
        result_queue.put(frame)
        time.sleep(0.04)


def convert_data():
    logger.debug("Entering")
    while receiver.is_alive():
        hw_stream()
        sw_stream()
        stopped_stream()

    logger.debug("Leaving")


def start_threads() -> None:
    logger.debug("Entering")
    global sender
    global receiver
    global converter
    logger.debug("Wait for mutex")
    locked = mutex_lock.acquire(timeout=0.1)
    if locked:
        logger.debug("Mutex aquired")
        _ = reset_fifos()
        sender = threading.Thread(target=send_data, name="sender")
        sender.start()
        receiver = threading.Thread(target=receive_data, name="receiver")
        receiver.start()
        converter = threading.Thread(target=convert_data, name="converter")
        converter.start()
    else:
        logger.debug("No Mutex aquired")
    logger.debug("Leaving")


def stop_threads() -> None:
    logger.debug("Entering")
    stop_producer.set()
    # converter only stops once producer and consumer is stopped so we only need to wait for the converter
    converter.join()
    flush_queues()
    benchmark_info.reset()

    logger.debug("Releasing mutex")
    mutex_lock.release()
    logger.debug("Leaving")


def flush_queues() -> None:
    logger.debug("Entering")
    while not send_queue.empty():
        _ = send_queue.get()

    # Empty queues to have a clean slate
    while not receive_queue.empty():
        _ = receive_queue.get()

    while not result_queue.empty():
        _ = result_queue.get()

    logger.debug("Leaving")


def gen_frames() -> Generator[Any, Any, Any]:  # pyright: ignore [reportExplicitAny]
    logger.debug("Entering")
    stop_producer.clear()
    start_threads()
    loop_timer = Timer("benchmark")
    count_bak = count

    while GlobalState.get_current_model() == Model.ONE_D_FFT.value:
        try:
            frame = result_queue.get_nowait()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            # calculate speed for hw
            count_current = count
            benchmark_info.fps = int(1 / loop_timer.duration() * (count_current - count_bak))
            count_bak = count_current

        except queue.Empty:
            time.sleep(0.001)
            continue

    stop_threads()
    logger.debug("Leaving")
