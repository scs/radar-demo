# pyright: reportAny=false
import ctypes
import logging
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


def send_1d_fft_data(data: NDArray[np.int16], data_size_in_bytes: int, batch_size: int) -> None:
    if STATIC_CONFIG.versal_lib:
        STATIC_CONFIG.versal_lib.send_1d_fft_data(
            data.ctypes,
            data_size_in_bytes,
            batch_size,
        )


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


def setup_plot(samples: int, amplitude: int) -> tuple[Figure, Line2D, NDArray[np.signedinteger[Any]], float]:
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


def gen_frames() -> Generator[Any, Any, Any]:
    amplitude = 32
    samples = 512
    scale = samples * amplitude
    fig, line, x, maxval = setup_plot(samples, amplitude)
    timer = Timer("benchmark")

    while GlobalState.get_current_model() == Model.ONE_D_FFT.value:

        if GlobalState.is_stopped():
            benchmark_info.reset()
            frame = STATIC_CONFIG.stopped_buf
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)
        else:
            signal_timer = Timer(name="signal generation")
            signal = get_current_signal(samples, amplitude)
            signal_timer.log_time()
            if GlobalState.use_hw():
                data_arangement_timer = Timer(name="Data Arangement")
                batch_size = GlobalState.get_current_batch_size()
                hw_data = np.zeros((samples, 2))
                hw_data[..., 0] = signal.real
                hw_data[..., 1] = signal.imag
                hw_data = hw_data.astype(np.int16)
                data_arangement_timer.log_time()
                send_timer = Timer(name="PCIe Send")
                send_1d_fft_data(
                    hw_data,
                    2 * samples * ctypes.sizeof(ctypes.c_int16),
                    batch_size,
                )
                send_timer.log_time()
                receive_timer = Timer(name="PCIe Receive")
                fft_data = np.zeros((samples, 2)).astype(np.int16)
                receive_1d_fft_results(fft_data, 2 * samples * ctypes.sizeof(ctypes.c_int16))
                receive_timer.log_time()
                plot_timer = Timer(name="Plot")
                real = fft_data[..., 0].astype(float)
                imag = fft_data[..., 1].astype(float)
                amp = np.sqrt(real**2 + imag**2)
                amp = amp / np.max(amp) * maxval
                line.set_data(x, amp)
                plot_timer.log_time()
            else:
                fft_data = np.fft.fft(signal, samples, norm="forward")
                fft_data = np.abs(fft_data) * scale / amplitude
                fft_data = fft_data / np.max(fft_data) * maxval
                line.set_data(x, np.abs(fft_data))

            buf = BytesIO()
            fig.savefig(buf, format="JPEG")

            frame = buf.getbuffer()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            benchmark_info.fps = int(1 / timer.duration())
