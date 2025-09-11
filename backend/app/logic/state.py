# pyright: reportAny=false
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from app.logic.config import STATIC_CONFIG
from app.logic.logging import LogLevel, get_logger
from app.logic.model import Model
from app.logic.settings import ComputePlatform, Settings, benchmark_settings, radar_settings
from position import compute_position

logger = get_logger(__name__, LogLevel.WARNING)


class RunningState(Enum):
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class PageState(Enum):
    LEAVING = "LEAVING"
    LEFT = "LEFT"
    ENTERED = "ENTERED"


@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {
            "w": self.w,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass
class Position:
    x: float
    y: float
    z: float

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


@dataclass
class ModelPosition:
    step: int
    position: Position
    orientation: Quaternion
    velocity: float

    def to_dict(self) -> dict[str, dict[str, float] | int | float]:
        return {
            "step": self.step,
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
            "velocity": self.velocity,
        }


class ModelPositionCollection:
    def __init__(self, scene_file: Path) -> None:
        if scene_file.is_file():
            data = {}
            with open(scene_file) as stats:
                data = json.load(stats)

            self.collection: list[ModelPosition] = [
                ModelPosition(
                    entry["step"],
                    Position(entry["target_x"], entry["target_y"], entry["target_z"]),
                    Quaternion(
                        entry["target_quaternion_w"],
                        entry["target_quaternion_x"],
                        entry["target_quaternion_y"],
                        entry["target_quaternion_z"],
                    ),
                    entry["projected_velocity"],
                )
                for entry in data
            ]

    def __getitem__(self, step: int) -> dict[str, dict[str, float] | int | float]:
        step = step % len(self.collection)
        return self.collection[step].to_dict()


class ModelPositionCollectionCollection:
    def __init__(self, scene_files: list[Path]) -> None:
        self.collections: list[ModelPositionCollection] = [ModelPositionCollection(file) for file in scene_files]

    def __getitem__(self, step: int) -> list[dict[str, dict[str, float] | int | float]]:
        return [collection[step] for collection in self.collections]


class GlobalState:
    settings: Settings = benchmark_settings
    running_state: RunningState = RunningState.STOPPED
    model: Model = Model("NONE")
    current_steps: list[int] = [0, 0, 0, 0]

    positions: dict[Model, ModelPositionCollectionCollection] = {
        Model.ONE_D_FFT: ModelPositionCollectionCollection(
            [Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps3s.json")]
        ),
        Model.SHORT_RANGE: ModelPositionCollectionCollection(
            [Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps3s.json")]
        ),
        Model.QUAD_CORNER: ModelPositionCollectionCollection(
            [
                Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps3s.json"),
                Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps4s.json"),
                Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps6s.json"),
                Path("stimuli/radardemo_scene_1tx4rx1024rg512dp30fps12s.json"),
            ]
        ),
        Model.IMAGING: ModelPositionCollectionCollection(
            [Path("stimuli/radardemo_scene_4tx16rx1024rg512dp30fps3s.json")]
        ),
    }
    amplitudes: list[tuple[int, int, int]] = [(5, 10, 0)] * 4
    offsets: list[tuple[int, int, int]] = [(0, 0, 0)] * 4

    @classmethod
    def to_dict(cls):

        if cls.model not in [Model.NONE]:
            current_positions = cls.positions.get(cls.model, {})[cls.current_steps[3]]
        else:
            current_positions = {}

        return {
            "settings": cls.settings.to_dict(),
            "runningState": cls.running_state.value,
            "model": cls.model.value,
            "current_steps": cls.current_steps,
            "current_positions": current_positions,
            "path": cls.get_current_path(),
            "path_scale": cls.get_path_scale(),
        }

    @classmethod
    def init_state(cls, model: str | None):
        if model is None:
            model = "NONE"
        cls.model = Model(model)
        cls.settings = benchmark_settings if cls.model == Model.ONE_D_FFT else radar_settings
        STATIC_CONFIG.probe_hw()
        if STATIC_CONFIG.versal_lib:
            cls.settings.set_device(ComputePlatform.VE2102)
        else:
            cls.settings.disable_hw()

        cls.running_state = RunningState.STOPPED
        cls.page_state: PageState = PageState.LEFT
        cls.current_steps = [0, 0, 0, 0]

        cls.current_positions = np.vectorize(compute_position, signature="(),(n),(n)->()")(
            [0, 0, 0, 0], cls.amplitudes, cls.offsets
        ).tolist()

    @classmethod
    def get_current_steps(cls) -> list[int]:
        return GlobalState.current_steps

    @classmethod
    def get_current_path(cls) -> list[dict[str, float]]:
        fps = STATIC_CONFIG.frame_rate_per_second
        looptime = STATIC_CONFIG.period_in_seconds[0]
        if cls.model == Model.IMAGING:
            rx = 16
            tx = 4
        else:
            rx = 4
            tx = 1

        path = Path(f"stimuli/radardemo_path_{tx}tx{rx}rx1024rg512dp{fps}fps{looptime}s.json")
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"Path file {path} not found.")
        return []

    @classmethod
    def get_path_scale(cls) -> tuple[float, float, float]:
        if cls.model == Model.IMAGING:
            return (0.3, 0.3, 0.3)
        return (2.0, 2.0, 2.0)

    @classmethod
    def set_steps(cls, steps: list[int]):
        GlobalState.current_steps = steps

    @classmethod
    def get_current_model(cls) -> Model:
        return GlobalState.model

    @classmethod
    def get_current_running_state(cls) -> RunningState:
        return GlobalState.running_state

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
    def cfar_enabled(cls) -> bool:
        return cls.get_current_settings().get_cfar_enabled()

    @classmethod
    def parallel_10x_enabled(cls) -> bool:
        return cls.get_current_settings().get_parallel_10x_enabled()

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
    def use_sw(cls) -> bool:
        return cls.get_current_device() == ComputePlatform.PC_EMULATION.value

    @classmethod
    def has_hw(cls) -> bool:
        return STATIC_CONFIG.versal_lib is not None

    @classmethod
    def is_running(cls) -> bool:
        return GlobalState.running_state != RunningState.STOPPED

    @classmethod
    def is_stopped(cls) -> bool:
        return GlobalState.running_state == RunningState.STOPPED

    @classmethod
    def is_stopping(cls) -> bool:
        return GlobalState.running_state == RunningState.STOPPING

    @classmethod
    def set_stopping(cls) -> None:
        if GlobalState.running_state == RunningState.RUNNING:
            GlobalState.running_state = RunningState.STOPPING

    @classmethod
    def set_running(cls) -> None:
        if GlobalState.running_state == RunningState.STOPPED:
            GlobalState.running_state = RunningState.RUNNING

    @classmethod
    def set_stopped(cls) -> None:
        GlobalState.running_state = RunningState.STOPPED

    @classmethod
    def get_current_state(cls):
        return GlobalState.to_dict()

    @classmethod
    def update_settings(cls, id: int, selectedSetting: dict[str, str]) -> None:
        if GlobalState.is_stopped():
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
