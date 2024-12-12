from dataclasses import dataclass
from enum import Enum


class ComputePlatform(Enum):
    PC_EMULATION = "PC Emulation"
    VE2302 = "Versal Edge VE2302"
    VE2102 = "Versal Edge VE2102"


def get_number_of_ai_elements(device: str) -> int:
    match device:
        case ComputePlatform.VE2102.value:
            return 12
        case ComputePlatform.VE2302.value:
            return 34
        case ComputePlatform.PC_EMULATION.value:
            return 0
        case _:
            return 0


@dataclass
class ValueMap:
    label: str
    availableOptions: list[str]

    def to_dict(self):
        return {"label": self.label, "availableOptions": self.availableOptions}


class SettingLabel(Enum):
    DEVICE = "Device"
    RANGE_FFT = "Range FFT"
    DOPPLER_FFT = "Doppler FFT"
    BATCH_SIZE = "Batch Size"


@dataclass
class Selected:
    type: str
    option: str | None

    def to_dict(self):
        if self.option:
            return {"type": self.type, "option": self.option}
        return {"type": self.type}

    def set_from_dict(self, sel: dict[str, str]) -> None:
        for key, value in sel.items():
            match key:
                case "type":
                    self.type = value
                case "option":
                    self.option = value
                case _:
                    ...


@dataclass
class Setting:
    label: SettingLabel
    valueMap: list[ValueMap]
    allOptions: list[str]
    selected: Selected
    id: int = 1
    enabled: bool = True

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label.value,
            "valueMap": [v.to_dict() for v in self.valueMap],
            "allOptions": self.allOptions,
            "selected": self.selected.to_dict(),
        }


class Settings:

    def __init__(self, name: str) -> None:
        self.settings: list[Setting] = []
        self.uniq_id: int = 1
        self.name: str = name
        self.index: int = 0

    def add_setting(self, setting: Setting) -> None:
        setting.id = self.uniq_id
        self.uniq_id += 1
        self.settings.append(setting)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        try:
            result = self.settings[self.index]
        except IndexError:
            self.index = 0
            raise StopIteration
        self.index += 1
        return result

    def to_dict(self):
        self.check_settings()
        return [setting.to_dict() for setting in self.settings if setting.enabled]

    def get_selected(self, label: SettingLabel) -> Selected | None:
        for setting in self.settings:
            if setting.label == label:
                return setting.selected
        return None

    def get_selected_option(self, label: SettingLabel) -> str | None:
        selected = self.get_selected(label)
        if selected and selected.option:
            return selected.option
        return None

    def get_device(self) -> str:
        if selected := self.get_selected(SettingLabel.DEVICE):
            return selected.type
        return ""

    def set_device(self, device: ComputePlatform) -> None:
        for setting in self.settings:
            if setting.label == SettingLabel.DEVICE:
                setting.selected.type = device.value

    def get_batch_size(self) -> int:
        if selected := self.get_selected(SettingLabel.BATCH_SIZE):
            return int(selected.type)
        return 1

    def check_settings(self):
        device = self.get_device()
        if device == ComputePlatform.PC_EMULATION.value:
            self.disable_batch_size()
        else:
            self.enable_batch_size()

    def disable_batch_size(self) -> None:
        for setting in self.settings:
            if setting.label == SettingLabel.BATCH_SIZE:
                setting.enabled = False

    def enable_batch_size(self) -> None:
        for setting in self.settings:
            if setting.label == SettingLabel.BATCH_SIZE:
                setting.enabled = True


device_setting = Setting(
    id=0,
    label=SettingLabel.DEVICE,
    valueMap=[
        ValueMap(label=ComputePlatform.VE2302.value, availableOptions=[]),
        ValueMap(label=ComputePlatform.VE2102.value, availableOptions=[]),
        ValueMap(label=ComputePlatform.PC_EMULATION.value, availableOptions=[]),
    ],
    allOptions=[],
    selected=Selected(type=ComputePlatform.PC_EMULATION.value, option=None),
)

one_d_fft = Setting(
    label=SettingLabel.RANGE_FFT,
    valueMap=[ValueMap("INT 16", ["512"]), ValueMap("INT 32", []), ValueMap("Brain Float 16", [])],
    allOptions=["128", "256", "512", "1024"],
    selected=Selected(type="INT 16", option="512"),
)

range_fft_setting = Setting(
    label=SettingLabel.RANGE_FFT,
    valueMap=[ValueMap("INT 16", ["1024"]), ValueMap("INT 32", []), ValueMap("Brain Float 16", [])],
    allOptions=["128", "256", "512", "1024"],
    selected=Selected(type="INT 16", option="1024"),
)

doppler_fft_setting = Setting(
    label=SettingLabel.DOPPLER_FFT,
    valueMap=[ValueMap("INT 16", ["512"]), ValueMap("INT 32", []), ValueMap("Brain Float 16", [])],
    allOptions=["128", "256", "512", "1024"],
    selected=Selected(type="INT 16", option="512"),
)

batch_size_setting = Setting(
    label=SettingLabel.BATCH_SIZE,
    valueMap=[
        ValueMap(label="1", availableOptions=[]),
        ValueMap(label="128", availableOptions=[]),
        ValueMap(label="512", availableOptions=[]),
        ValueMap(label="1024", availableOptions=[]),
        ValueMap(label="2048", availableOptions=[]),
        ValueMap(label="4096", availableOptions=[]),
        ValueMap(label="8192", availableOptions=[]),
        ValueMap(label="16384", availableOptions=[]),
    ],
    allOptions=[],
    selected=Selected(type="1024", option=None),
)

benchmark_settings = Settings(name="benchmark")
benchmark_settings.add_setting(one_d_fft)
benchmark_settings.add_setting(batch_size_setting)
benchmark_settings.add_setting(device_setting)

radar_settings = Settings(name="radar")
radar_settings.add_setting(range_fft_setting)
radar_settings.add_setting(doppler_fft_setting)
radar_settings.add_setting(device_setting)
