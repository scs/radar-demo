# pyright: reportAny=false
import ctypes
import queue
import threading
import time
from collections.abc import Generator
from io import BytesIO
from typing import Any, TypeVar

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from app.logic.config import STATIC_CONFIG
from app.logic.flush_card import buffer_status
from app.logic.logging import LogLevel, get_logger
from app.logic.output_exception import InputFull, OutputEmpty
from app.logic.state import GlobalState
from app.logic.status import benchmark_info
from app.logic.timer import Timer

matplotlib.use("agg")


#######################################################################################################################
# Module Global Variables
#

logger = get_logger(__name__, LogLevel.WARNING)

receive_queue: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=7)
result_queue = queue.Queue(maxsize=2)
stop_producer = threading.Event()
stop_receiver = threading.Event()
stop_converter = threading.Event()

send_count = threading.Semaphore(0)

AMPLITUDE = 32
SAMPLES = 512
SCALE = SAMPLES * AMPLITUDE
count = 0


#
#######################################################################################################################


class PlotSetup:
    fig: Figure | None = None
    line: Line2D | None = None
    x: NDArray[np.int32] | None = None
    maxval: float = 0

    @classmethod
    def setup_plot(cls, signal: NDArray[np.int32]) -> None:
        x = np.arange(0, len(signal)).astype(np.int32)
        fft_data = np.fft.fft(signal, len(signal), norm="forward")
        fft_data = fft_data * len(signal)
        _ = plt.ioff()
        fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
        absval = np.abs(fft_data)
        [line] = ax.plot(x, absval)
        line.set_data(x, absval)
        _ = ax.axis("off")
        PlotSetup.fig = fig
        PlotSetup.line = line
        PlotSetup.maxval = np.max(absval)
        PlotSetup.x = x


def get_current_phase(samples: int) -> NDArray[np.float32]:
    current_step = GlobalState.get_current_steps()[0]
    border = 20
    half_pattern = [*range(0, int(STATIC_CONFIG.number_of_steps_in_period[0] / 2))]
    pattern = half_pattern + half_pattern[::-1]
    position = pattern[current_step]
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


fig, line, x, maxval = setup_plot(SAMPLES, AMPLITUDE)


def send_1d_fft_data(data: NDArray[np.int16], data_size_in_bytes: int, batch_size: int) -> int:
    logger.debug("Entering")
    err = 1
    if STATIC_CONFIG.versal_lib:
        if STATIC_CONFIG.versal_lib.input_ready():
            err: int = STATIC_CONFIG.versal_lib.send_1d_fft_data(
                data.ctypes,
                data_size_in_bytes,
                batch_size,
            )
            if err == 0:
                send_count.release()
        else:
            # logger.warning("No empty input buffers available")
            raise InputFull()
    logger.debug("Leaving")
    return err


def receive_1d_fft_results(data: NDArray[np.int16], size: int) -> None:
    logger.debug("Entering")
    if STATIC_CONFIG.versal_lib:
        output_ready: int = STATIC_CONFIG.versal_lib.output_ready()
        if output_ready == 0:
            # logger.warning("No occupied output buffers available")
            raise OutputEmpty()
        STATIC_CONFIG.versal_lib.receive_1d_fft_results(data.ctypes, size)
    logger.debug("Leaving")


def send_data():
    logger.debug("Entering")
    if GlobalState.has_hw():
        while not stop_producer.is_set():
            if GlobalState.use_hw() and GlobalState.is_running():
                signal = get_current_signal(SAMPLES, AMPLITUDE)
                data_arangement_timer = Timer(name="Data Arangement")
                batch_size = GlobalState.get_current_batch_size()
                hw_data = np.zeros((SAMPLES, 2))
                hw_data[..., 0] = signal.real
                hw_data[..., 1] = signal.imag
                hw_data = hw_data.astype(np.int16)
                data_arangement_timer.log_time()
                send_timer = Timer(name="PCIe Send")
                try:
                    err = send_1d_fft_data(
                        hw_data,
                        2 * SAMPLES * ctypes.sizeof(ctypes.c_int16),
                        batch_size,
                    )

                    if err != 0:
                        logger.error("Failed to send 1D FFT Data")
                except InputFull:
                    time.sleep(0.01)
                send_timer.log_time()
            else:
                time.sleep(0.1)
    else:
        while not stop_producer.is_set():
            time.sleep(0.1)
    stop_receiver.set()
    logger.debug("Leaving")


def receive_result() -> NDArray[np.int16]:
    logger.debug("Entering")
    fft_data: NDArray[np.int16] = np.zeros((SAMPLES, 2)).astype(np.int16)
    receive_1d_fft_results(fft_data, 2 * SAMPLES * ctypes.sizeof(ctypes.c_int16))
    logger.debug("Leaving")
    return fft_data


def receive_data():
    logger.debug("Entering")
    global count
    while not stop_receiver.is_set():
        if GlobalState.has_hw():
            try:
                fft_data = receive_result()
                _ = send_count.acquire()
                count = count + 1
                if not receive_queue.full():
                    receive_queue.put(fft_data)
            except OutputEmpty:
                time.sleep(0.01)
        else:
            time.sleep(0.1)

    while send_count.acquire(blocking=False):
        received = False
        while not received:
            try:
                _ = receive_result()
                received = True
            except OutputEmpty:
                time.sleep(0.01)

    buffer_status(LogLevel.INFO)

    stop_converter.set()
    logger.debug("Leaving")


def create_frame(fft_data: NDArray[np.float32]):
    fft_data *= maxval
    line.set_data(x, np.abs(fft_data))

    buf = BytesIO()
    fig.savefig(buf, format="JPEG")

    frame = buf.getbuffer()
    return frame


def hw_stream():
    while not stop_converter.is_set() and not GlobalState.is_stopped() and GlobalState.use_hw():
        fft_data = None
        while not receive_queue.empty():
            fft_data = receive_queue.get()
        if fft_data is not None and not result_queue.full():
            plot_timer = Timer(name="Plot")
            real: NDArray[np.float32] = fft_data[..., 0].astype(np.float32)
            imag: NDArray[np.float32] = fft_data[..., 1].astype(np.float32)
            amp: NDArray[np.float32] = np.sqrt(real**2 + imag**2)
            norm_amp: NDArray[np.float32] = amp / np.max(amp)
            plot_timer.log_time()
            result_queue.put(create_frame(norm_amp))


def sw_stream():
    logger.debug("Entering")
    while not stop_converter.is_set() and not GlobalState.is_stopped() and not GlobalState.use_hw():
        signal_timer = Timer(name="signal generation")
        signal = get_current_signal(SAMPLES, AMPLITUDE)
        signal_timer.log_time()
        fft_timer = Timer("NP fft timer")
        fft_data = np.fft.fft(signal, SAMPLES, norm="forward")
        benchmark_info.set_ffts_emulation(int(1 / fft_timer.duration()))
        fft_data = np.abs(fft_data) * SAMPLES
        fft_data = fft_data / np.max(fft_data)
        if not result_queue.full():
            result_queue.put(create_frame(fft_data))
    logger.debug("Leaving")


def stopped_stream():
    logger.debug("Entering")
    flush_queue(result_queue)  # pyright: ignore [reportUnknownArgumentType]
    flush_queue(receive_queue)
    while not stop_converter.is_set() and GlobalState.is_stopped():
        flush_queue(receive_queue)
        benchmark_info.reset_frame_rate
        frame = STATIC_CONFIG.stopped_buf
        if not result_queue.full():
            result_queue.put(frame)
        time.sleep(0.04)
    logger.debug("Entering")


def convert_data():
    logger.debug("Entering")
    while not stop_converter.is_set():
        hw_stream()
        sw_stream()
        stopped_stream()

    flush_queue(receive_queue)
    flush_queue(result_queue)  # pyright: ignore [reportUnknownArgumentType]
    logger.debug("Leaving")


def start_threads() -> None:
    logger.debug("Entering")
    global sender
    global receiver
    global converter
    logger.debug("Wait for mutex")
    stop_producer.clear()
    stop_receiver.clear()
    stop_converter.clear()
    flush_queues()
    logger.debug("Mutex aquired")
    sender = threading.Thread(target=send_data, name="sender")
    sender.start()
    receiver = threading.Thread(target=receive_data, name="receiver")
    receiver.start()
    converter = threading.Thread(target=convert_data, name="converter")
    converter.start()
    logger.debug("Leaving")


def stop_threads() -> None:
    logger.debug("Entering")
    stop_producer.set()
    sender.join()
    receiver.join()
    converter.join()
    flush_queues()
    benchmark_info.reset()
    logger.debug("Leaving")


T = TypeVar("T")


def flush_queue(q: queue.Queue[T]):
    while not q.empty():
        _ = q.get()


def flush_queues() -> None:
    logger.debug("Entering")
    flush_queue(receive_queue)
    flush_queue(result_queue)  # pyright: ignore [reportUnknownArgumentType]
    # _ = flush_card(400)
    logger.debug("Leaving")


def gen_frames() -> Generator[Any, Any, Any]:  # pyright: ignore [reportExplicitAny]
    logger.debug("Entering")
    start_threads()
    loop_timer = Timer("benchmark")
    count_bak = count

    while not GlobalState.stop_producer.is_set():
        logger.debug(f"Model == {GlobalState.get_current_model().value}")
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
    GlobalState.stop_producer.clear()
