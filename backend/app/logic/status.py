from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from flask import Response, jsonify
from numpy.lib import math
from numpy.typing import NDArray

from app.logic.model import Model
from app.logic.settings import ComputePlatform, SettingLabel, Settings, get_number_of_ai_elements
from app.logic.state import GlobalState

WIN_SIZE = 20


def get_info():
    settings = GlobalState.get_current_settings()
    if GlobalState.get_current_model() == Model.ONE_D_FFT.value:
        return benchmark_info.get_info(settings)
    else:
        return range_doppler_info.get_info(settings)


def gen_radar_data() -> Response:
    return jsonify(get_info())


@dataclass
class HwInfo(ABC):
    frame_rate: NDArray[np.int32] = field(default_factory=lambda: np.zeros(WIN_SIZE).astype(np.int32))
    power: float = 0
    num_aie_used: int = 1

    def reset(self) -> None:
        self.frame_rate = np.zeros(WIN_SIZE).astype(np.int32)
        self.power = 0

    @property
    def fps(self) -> int:
        frame_rate = np.mean(self.frame_rate)
        return int(frame_rate)

    @fps.setter
    def fps(self, value: int):
        self.frame_rate = np.append(self.frame_rate, value)
        self.frame_rate = np.delete(self.frame_rate, 0)

    @property
    def reset_frame_rate(self):
        self.frame_rate = np.zeros(WIN_SIZE).astype(np.int32)

    @property
    def watt(self) -> str:
        if self.power == 0:
            return "-"
        else:
            return f"{self.power:.2f} W"

    def number_of_aie_used(self, settings: Settings) -> str:
        device = settings.get_device()
        available_aies = get_number_of_ai_elements(device)
        if available_aies > 0:
            return f"{self.num_aie_used}/{available_aies}"
        else:
            return "-"

    @abstractmethod
    def aie_load_percent(self, settings: Settings) -> str | None: ...

    @abstractmethod
    def fft_per_sec(self, settings: Settings) -> str: ...

    def int2str(self, value: int) -> str:
        if value == 0:
            return "-"
        else:
            return str(value)

    def get_info(self, settings: Settings):
        data: list[dict[str, str]] = [
            {
                "label": "FPS",
                "value": f"{self.int2str(self.fps)} fps",
            },
            {"label": "Power", "value": f"{self.watt}"},
            {
                "label": "AIE used",
                "value": f"{self.number_of_aie_used(settings)}",
            },
            {"label": "AIE utilisation", "value": f"{self.aie_load_percent(settings)}"},
            {"label": "FFT/sec", "value": self.fft_per_sec(settings)},
        ]

        return data


@dataclass
class Fft1DInfo(HwInfo):
    num_aie_used: int = 1

    def aie_load_percent(self, settings: Settings) -> str | None:  # pyright: ignore [reportImplicitOverride]
        map_min_time = {
            512: 0.0000008,
            1024: 0.0000016,
        }

        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return "-"
        else:
            option = int(settings.get_selected_option(SettingLabel.RANGE_FFT) or 0)
            load = -1
            if option in map_min_time:
                min_time = map_min_time[option]
                fps = np.mean(self.frame_rate)
                batch_size = GlobalState.get_current_batch_size()
                load = min_time * fps * batch_size * 100

            if load == -1:
                retval = "nil"
            else:
                if load == 0:
                    return "-"
                retval_format = '"{' + f"load:.{int(max(1+math.log10(1/load), 0))}f" + '}%"'
                retval = eval(f"f{retval_format}")  # pyright: ignore [reportAny]
            return retval

    def fft_per_sec_int(self, settings: Settings) -> int:
        batch_size = GlobalState.get_current_batch_size()
        mean = np.mean(self.frame_rate)
        return int(batch_size * mean)

    def fft_per_sec(self, settings: Settings) -> str:  # pyright: ignore [reportImplicitOverride]
        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return str(self.fps)
        else:
            return f"{self.fft_per_sec_int(settings):,}"


@dataclass
class RangeDopplerInfo(HwInfo):
    num_aie_used: int = 2

    def aie_load_percent(self, settings: Settings) -> str | None:  # pyright: ignore [reportImplicitOverride]
        max_fft_per_sec = 1 / 0.0000008 + 1 / 0.0000016
        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return "-"
        else:
            load = -1
            load = self.fft_per_sec_int(settings) / max_fft_per_sec * 100

            if load == -1:
                retval = "nil"
            else:
                if load == 0:
                    return "-"
                retval_format = '"{' + f"load:.{int(max(1+math.log10(1/load), 0))}f" + '}%"'
                retval = eval(f"f{retval_format}")  # pyright: ignore [reportAny]
            return retval

    def fft_per_sec_int(self, settings: Settings) -> int:
        range_config = settings.get_selected_option(SettingLabel.RANGE_FFT) or 1
        doppler_config = settings.get_selected_option(SettingLabel.DOPPLER_FFT) or 1
        range_size = int(range_config)
        doppler_size = int(doppler_config)
        mean = np.mean(self.frame_rate)
        channels = 16
        if GlobalState.get_current_model() == Model.SHORT_RANGE.value:
            channels /= 4
        return int(channels * (range_size + doppler_size) * mean)

    def fft_per_sec(self, settings: Settings) -> str:  # pyright: ignore [reportImplicitOverride]
        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return "-"
        else:
            return f"{self.fft_per_sec_int(settings):,}"


benchmark_info = Fft1DInfo()
range_doppler_info = RangeDopplerInfo()
