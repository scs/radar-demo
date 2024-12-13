# pyright: reportAny=false
import json
import time
from enum import Enum

import numpy as np

from app.logic.config import STATIC_CONFIG
from app.logic.model import Model
from app.logic.settings import ComputePlatform, Settings, benchmark_settings, radar_settings
from position import compute_position


class RunningState(Enum):
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


def compute_position_old(phase: float) -> dict[str, int]:
    return {
        "x": (np.cos(phase) * np.sin(phase)) / (np.sin(phase) ** 2 + 1) * 10,
        "y": (np.cos(phase) / (np.sin(phase) ** 2 + 1)) * 10,
        "z": 0,
    }


class GlobalState:
    settings: Settings = benchmark_settings
    runningState: RunningState = RunningState.STOPPED
    model: str | None = None
    current_steps: list[int] = [0, 0, 0, 0]
    current_positions: tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]] = (
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 0.0},
    )
    amplitudes: list[tuple[int, int, int]] = [(5, 10, 0)] * 4
    offsets: list[tuple[int, int, int]] = [(0, 0, 0)] * 4

    @classmethod
    def to_dict(cls):
        return {
            "settings": cls.settings.to_dict(),
            "runningState": cls.runningState.value,
            "model": cls.model,
            "current_steps": cls.current_steps,
            "current_positions": cls.current_positions,
        }

    @classmethod
    def init_state(cls, model: str | None):
        cls.settings = benchmark_settings if model == Model.ONE_D_FFT.value else radar_settings
        if STATIC_CONFIG.versal_lib:
            cls.settings.set_device(ComputePlatform.VE2302)
        else:
            cls.settings.disable_hw()

        cls.runningState = RunningState.STOPPED
        cls.model = model
        cls.current_steps = [0, 0, 0, 0]

        cls.current_positions = np.vectorize(compute_position, signature="(),(n),(n)->()")(
            [0, 0, 0, 0], cls.amplitudes, cls.offsets
        ).tolist()

    @classmethod
    def get_current_steps(cls) -> list[int]:
        return GlobalState.current_steps

    @classmethod
    def set_steps(cls, steps: list[int]):
        GlobalState.current_steps = steps

    @classmethod
    def get_current_model(cls) -> str | None:
        return GlobalState.model

    @classmethod
    def get_current_running_state(cls) -> RunningState:
        return GlobalState.runningState

    @classmethod
    def get_current_positions(cls) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        return GlobalState.current_positions

    @classmethod
    def set_positions(
        cls, positions: tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]
    ) -> None:
        GlobalState.current_positions = positions

    @classmethod
    def get_current_settings(cls) -> Settings:
        return GlobalState.settings

    @classmethod
    def get_current_device(cls) -> str:
        return cls.get_current_settings().get_device()

    @classmethod
    def get_current_batch_size(cls) -> int:
        return cls.get_current_settings().get_batch_size()

    @classmethod
    def use_emulation(cls) -> bool:
        return cls.get_current_device() == ComputePlatform.PC_EMULATION.value

    @classmethod
    def use_hw(cls) -> bool:
        return cls.get_current_device() != ComputePlatform.PC_EMULATION.value and STATIC_CONFIG.versal_lib is not None

    @classmethod
    def has_hw(cls) -> bool:
        return STATIC_CONFIG.versal_lib is not None

    @classmethod
    def is_running(cls) -> bool:
        return GlobalState.runningState != RunningState.STOPPED

    @classmethod
    def is_stopped(cls) -> bool:
        return GlobalState.runningState == RunningState.STOPPED

    @classmethod
    def is_stopping(cls) -> bool:
        return GlobalState.runningState == RunningState.STOPPING

    @classmethod
    def set_stopping(cls) -> None:
        if GlobalState.runningState == RunningState.RUNNING:
            GlobalState.runningState = RunningState.STOPPING

    @classmethod
    def set_running(cls) -> None:
        if GlobalState.runningState == RunningState.STOPPED:
            GlobalState.runningState = RunningState.RUNNING

    @classmethod
    def set_stopped(cls) -> None:
        GlobalState.runningState = RunningState.STOPPED

    @classmethod
    def get_current_state(cls):
        return GlobalState.to_dict()

    @classmethod
    def update_settings(cls, id: int, selectedSetting: dict[str, str]) -> None:
        for setting in GlobalState.settings:
            if setting.id == id:
                setting.selected.set_from_dict(selectedSetting)
                break

    @classmethod
    def increment_step(cls) -> None:
        steps_array = np.array(cls.get_current_steps())
        steps_array += 1
        steps_array %= STATIC_CONFIG.number_of_steps_in_period
        cls.set_steps(steps_array.tolist())

    @classmethod
    def gen_frame_number(cls) -> list[int]:
        cls.increment_step()
        t = np.array(cls.get_current_steps())
        positions = np.vectorize(compute_position, signature="(),(n),(n)->()")(
            (2 * np.pi * t / STATIC_CONFIG.number_of_steps_in_period), cls.amplitudes, cls.offsets
        ).tolist()
        cls.set_positions(positions)
        time.sleep(1 / STATIC_CONFIG.frame_rate_per_second)
        return t.tolist()

    @classmethod
    def gen_frame_number_response(cls):
        while True:
            frame_number = cls.gen_frame_number()
            completion = frame_number / STATIC_CONFIG.number_of_steps_in_period
            if cls.is_stopping():
                if frame_number[0] == 0:
                    cls.set_stopped()

            data = {
                "frameNumber": frame_number,
                "periodCompletion": completion.tolist(),
                "positions": cls.get_current_positions(),
                "runningState": cls.get_current_running_state().value,
            }
            yield f"data: {json.dumps(data)}\n\n"

            if cls.is_stopped():
                break
