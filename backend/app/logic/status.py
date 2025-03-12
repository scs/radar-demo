from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from flask import Response, jsonify
from numpy.lib import math

from app.logic.config import STATIC_CONFIG
from app.logic.model import Model
from app.logic.settings import ComputePlatform, SettingLabel, Settings, get_number_of_ai_elements
from app.logic.state import GlobalState

WIN_SIZE = 20


def get_info():
    settings = GlobalState.get_current_settings()
    if GlobalState.get_current_model() == Model.ONE_D_FFT:
        if STATIC_CONFIG.versal_lib:
            benchmark_info.temperature = STATIC_CONFIG.versal_lib.get_temp_val() / 128.0
            benchmark_info.power = STATIC_CONFIG.versal_lib.get_power_val() / 1000.0
        return benchmark_info.get_info(settings)
    else:
        if STATIC_CONFIG.versal_lib:
            range_doppler_info.temperature = STATIC_CONFIG.versal_lib.get_temp_val() / 128.0
            range_doppler_info.power = STATIC_CONFIG.versal_lib.get_power_val() / 1000.0
        return range_doppler_info.get_info(settings)


def gen_radar_data() -> Response:
    return jsonify(get_info())


@dataclass
class HwInfo(ABC):
    frame_rate: list[int] = field(default_factory=lambda: [0])
    power: float = 0
    temperature: float = 0
    num_aie_used: int = 1

    def reset(self) -> None:
        self.frame_rate = [0]
        self.power = 0

    @property
    def fps(self) -> int:
        frame_rate = np.mean(self.frame_rate)
        return int(frame_rate)

    @fps.setter
    def fps(self, value: int):
        self.frame_rate.append(value)
        if len(self.frame_rate) > WIN_SIZE:
            _ = self.frame_rate.pop(0)

    @property
    def reset_frame_rate(self):
        self.frame_rate = [0]

    @property
    def watt(self) -> str:
        if self.power == 0:
            return "-"
        else:
            return f"{self.power:.2f} W"

    @property
    def temp(self) -> str:
        if self.temperature == 0:
            return "-"
        else:
            return f"{self.temperature:.1f} °C"

    @abstractmethod
    def aie_usage(self, settings: Settings) -> list[int]: ...

    @abstractmethod
    def fft_per_sec_int(self, settings: Settings) -> int: ...

    @abstractmethod
    def fft_per_sec(self, settings: Settings) -> str: ...

    def int2str(self, value: int) -> str:
        if value == 0:
            return "-"
        else:
            return str(value)

    def pl_usage(
        self,
    ) -> int:
        if GlobalState.cfar_enabled():
            return 53
        return 52

    def ps_usage(self) -> int:
        return 3

    def get_info(self, settings: Settings):
        data: list[dict[str, str | list[int] | int]] = [
            {"label": "Power", "value": f"{self.watt}"},
            {"label": "Temp", "value": f"{self.temp}"},
        ]
        if settings.get_device() != ComputePlatform.PC_EMULATION.value:
            data.append({"label": "AIE", "value": self.aie_usage(settings)})
            data.append({"label": "PL", "value": self.pl_usage()})
            data.append({"label": "PS", "value": self.ps_usage()})
        return data

    def format_load(self, load: float) -> str:
        retval_format = '"{' + f"load:.{int(max(1+math.log10(1/load), 0))}f" + '}%"'
        retval: str = eval(f"f{retval_format}")
        return retval


@dataclass
class Fft1DInfo(HwInfo):
    num_aie_used: int = 1
    ffts_emulation: list[int] = field(default_factory=lambda: [0])

    def reset(self):  # pyright: ignore [reportImplicitOverride]
        self.frame_rate: list[int] = [0]
        self.power: float = 0
        self.ffts_emulation = [0]

    @property
    def reset_frame_rate(self):  # pyright: ignore [reportImplicitOverride]
        self.frame_rate = [0]
        self.ffts_emulation = [0]

    def set_ffts_emulation(self, value: int):
        self.ffts_emulation.append(value)
        if len(self.ffts_emulation) > 1000:
            _ = self.ffts_emulation.pop(0)

    def aie_usage(self, settings: Settings) -> list[int]:  # pyright: ignore [reportImplicitOverride]
        device = settings.get_device()
        available_aies = get_number_of_ai_elements(device)
        map_min_time = {
            512: 0.0000008,
            1024: 0.0000016,
        }

        device = settings.get_device()
        retval = [0] * available_aies
        if device == ComputePlatform.PC_EMULATION.value:
            return []
        else:
            option = int(settings.get_selected_option(SettingLabel.RANGE_FFT) or 0)
            load = -1
            if option in map_min_time:
                min_time = map_min_time[option]
                fps = np.mean(self.frame_rate)
                batch_size = GlobalState.get_current_batch_size()
                load = min_time * fps * batch_size * 100
                if GlobalState.parallel_10x_enabled():
                    for i in range(10):
                        retval[i] = min(int(load), 100)
                else:
                    retval[0] = min(int(load), 100)

            return retval

    def fft_per_sec_int(self, settings: Settings) -> int:  # pyright: ignore [reportImplicitOverride]
        if settings.get_device() == ComputePlatform.PC_EMULATION.value:
            return int(np.mean(self.ffts_emulation))
        batch_size = GlobalState.get_current_batch_size()
        mean = np.mean(self.frame_rate)
        return int(batch_size * mean)

    def fft_per_sec(self, settings: Settings) -> str:  # pyright: ignore [reportImplicitOverride]
        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return f"{int(np.mean(self.ffts_emulation)):,}"
        else:
            if GlobalState.parallel_10x_enabled():
                return f"{self.fft_per_sec_int(settings) * 10:,}"
            else:
                return f"{self.fft_per_sec_int(settings):,}"

    def get_info(self, settings: Settings):  # pyright: ignore [reportImplicitOverride]
        data = super().get_info(settings)
        data.append(
            {"label": "FFT/sec", "value": self.fft_per_sec(settings)},
        )
        return data


@dataclass
class RangeDopplerInfo(HwInfo):
    num_aie_used: int = 2

    def aie_usage(self, settings: Settings) -> list[int]:  # pyright: ignore [reportImplicitOverride]

        doppler_max_fft_per_sec = 1 / 0.0000018
        range_max_fft_per_sec = 1 / 0.0000009
        device = settings.get_device()
        available_aies = get_number_of_ai_elements(device)
        if device == ComputePlatform.PC_EMULATION.value:
            load = []
        else:
            load = [0] * available_aies
            load[0] = min(int(self.range_fft_per_sec_int(settings) / range_max_fft_per_sec * 100), 100)
            load[1] = min(int(self.doppler_fft_per_sec_int(settings) / doppler_max_fft_per_sec * 100), 100)
        return load

    def generic_fft_per_sec_int(self, fft_size: int) -> int:
        mean = np.mean(self.frame_rate)
        channels = 16
        if GlobalState.get_current_model() == Model.SHORT_RANGE:
            channels /= 4
        return int(channels * fft_size * mean)

    def range_fft_per_sec_int(self, settings: Settings) -> int:
        range_config = settings.get_selected_option(SettingLabel.RANGE_FFT) or 1
        range_size = int(range_config)
        return self.generic_fft_per_sec_int(range_size)

    def doppler_fft_per_sec_int(self, settings: Settings) -> int:
        doppler_config = settings.get_selected_option(SettingLabel.DOPPLER_FFT) or 1
        doppler_size = int(doppler_config)
        return self.generic_fft_per_sec_int(doppler_size)

    def fft_per_sec_int(self, settings: Settings) -> int:  # pyright: ignore [reportImplicitOverride]
        return self.range_fft_per_sec_int(settings) + self.doppler_fft_per_sec_int(settings)

    def fft_per_sec(self, settings: Settings) -> str:  # pyright: ignore [reportImplicitOverride]
        device = settings.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            return "-"
        else:
            return f"{self.fft_per_sec_int(settings):,}"

    def get_info(  # pyright: ignore [reportImplicitOverride]
        self, settings: Settings
    ) -> list[dict[str, str | list[int] | int]]:
        data = super().get_info(settings)
        data.append(
            {
                "label": "FPS",
                "value": f"{self.int2str(self.fps)} fps",
            }
        )
        data.append(
            {"label": "Doppler FFT/sec", "value": f"{self.doppler_fft_per_sec_int(settings):,}"},
        )
        data.append(
            {"label": "Range FFT/sec", "value": f"{self.range_fft_per_sec_int(settings):,}"},
        )
        return data


benchmark_info = Fft1DInfo()
range_doppler_info = RangeDopplerInfo()
