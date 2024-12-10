# pyright: reportAny=false
import ctypes
import logging
import queue
import threading
import time
from collections.abc import Generator
from io import BytesIO
from typing import Any

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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("radar demo app")

result_queue = queue.Queue(maxsize=7)
send_queue = queue.Queue(maxsize=7)
receive_queue = queue.Queue(maxsize=7)
stop_producer = threading.Event()
mutex_lock = threading.Lock()

AMPLITUDE = 32
SAMPLES = 512
SCALE = SAMPLES * AMPLITUDE
count = 0


def send_1d_fft_data(data: NDArray[np.int16], data_size_in_bytes: int, batch_size: int) -> int:
    err = 1
    if STATIC_CONFIG.versal_lib:
        err: int = STATIC_CONFIG.versal_lib.send_1d_fft_data(
            data.ctypes,
            data_size_in_bytes,
            batch_size,
        )
    return err


def receive_1d_fft_results(data: NDArray[np.int16], size: int) -> None:
    if STATIC_CONFIG.versal_lib:
        STATIC_CONFIG.versal_lib.receive_1d_fft_results(data.ctypes, size)


def get_current_phase(samples: int) -> NDArray[np.float32]:
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
    return phase


def get_current_signal(samples: int, amplitude: int) -> NDArray[np.int32]:
    phase = get_current_phase(samples)
    rand_noise = np.random.normal(scale=0.15, size=samples)
    signal = np.exp(1.0j * (phase + rand_noise)) * amplitude
    return signal


def setup_plot(samples: int, amplitude: int) -> tuple[Figure, Line2D, NDArray[np.int32], float]:
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
    return (fig, line, x, np.max(absval))


def reset_fifos() -> bool:
    if STATIC_CONFIG.versal_lib:
        err = STATIC_CONFIG.versal_lib.reset_hw()
        return err == 0
    return True


def send_data():
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

    logger.info("Stopping sender thread")


def receive_result():
    fft_data = np.zeros((SAMPLES, 2)).astype(np.int16)
    receive_1d_fft_results(fft_data, 2 * SAMPLES * ctypes.sizeof(ctypes.c_int16))
    receive_queue.put(fft_data)


def receive_data():
    global count
    timer = Timer(name="receive loop")
    while sender.is_alive():
        if GlobalState.use_hw():
            try:
                _ = send_queue.get_nowait()
                receive_result()
                duration = timer.duration()
                count = count + 1
            except queue.Empty:
                time.sleep(0.001)
                pass
        else:
            time.sleep(0.1)


def convert_data():
    while receiver.is_alive():
        if GlobalState.is_stopped():
            time.sleep(0.1)
        elif GlobalState.use_hw():
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
                result_queue.put(amp)
        else:
            signal_timer = Timer(name="signal generation")
            signal = get_current_signal(SAMPLES, AMPLITUDE)
            signal_timer.log_time()
            fft_timer = Timer("NP fft timer")
            fft_data = np.fft.fft(signal, SAMPLES, norm="forward")
            benchmark_info.set_ffts_emulation(int(1 / fft_timer.duration()))
            fft_data = np.abs(fft_data) * SAMPLES
            fft_data = fft_data / np.max(fft_data)
            result_queue.put(fft_data)


def start_threads() -> None:
    logger.info("Starting Threads")
    global sender
    global receiver
    global converter
    logger.info("Wait for mutex")
    locked = mutex_lock.acquire(timeout=0.1)
    if locked:
        logger.info("Mutex aquired")
        _ = reset_fifos()
        sender = threading.Thread(target=send_data, name="sender")
        sender.start()
        receiver = threading.Thread(target=receive_data, name="receiver")
        receiver.start()
        converter = threading.Thread(target=convert_data, name="converter")
        converter.start()


def stop_threads() -> None:
    logger.info("Stopping threads")
    stop_producer.set()
    # converter only stops once producer and consumer is stopped so we only need to wait for the converter
    converter.join()
    flush_queues()
    benchmark_info.reset()

    logger.info("Releasing mutex")
    mutex_lock.release()


def flush_queues() -> None:
    while not send_queue.empty():
        _ = send_queue.get()

    # Empty queues to have a clean slate
    while not receive_queue.empty():
        _ = receive_queue.get()

    while not result_queue.empty():
        _ = result_queue.get()


def gen_frames() -> Generator[Any, Any, Any]:  # pyright: ignore [reportExplicitAny]
    fig, line, x, maxval = setup_plot(SAMPLES, AMPLITUDE)
    start_threads()
    loop_timer = Timer("benchmark")
    count_bak = count
    while GlobalState.get_current_model() == Model.ONE_D_FFT.value:

        if GlobalState.is_stopped():
            flush_queues()
            benchmark_info.reset()
            frame = STATIC_CONFIG.stopped_buf
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)
        else:
            try:
                fft_data = result_queue.get_nowait()
                fft_data *= maxval
                line.set_data(x, np.abs(fft_data))  # pyright: ignore  [reportUnknownArgumentType]

                buf = BytesIO()
                fig.savefig(buf, format="JPEG")

                frame = buf.getbuffer()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                count_current = count
                benchmark_info.fps = int(1 / loop_timer.duration() * (count_current - count_bak))
                count_bak = count_current

            except queue.Empty:
                time.sleep(0.001)
                continue
    logger.info("Stopping Threads")
    stop_threads()
