import logging
from ctypes import CDLL, cdll
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageEnhance

from app.package import package_root

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("radar demo app")

ARTEFACTS = package_root / "artefacts"
STOPPED_IMG = ARTEFACTS / "stopped.jpeg"
DATA_SERVER = ARTEFACTS / "libdata_server.so"


@dataclass
class StaticConfig:
    video_dim: int = 1024
    export_results: bool = False
    frame_rate_per_second: int = 30
    iter: int = 0
    versal_lib: None | CDLL = None

    def __init__(self):
        self.period_in_seconds: NDArray[np.int32] = np.array([3, 4, 6, 12])
        self.static_stopped_image: Image.Image = Image.open(STOPPED_IMG)

        class LibOption(Enum):
            HW = 0
            EMULATE = 1

        lib_option = LibOption.HW.value

        if Path.exists(DATA_SERVER):
            self.versal_lib = cdll.LoadLibrary(str(DATA_SERVER))
            if self.versal_lib:
                return_code: int = self.versal_lib.init_server(lib_option)
                if return_code != lib_option:
                    logger.error(f"Failed to open Shared Object return code was {return_code} expected {lib_option}")
                    self.versal_lib = None

    @property
    def number_of_steps_in_period(self) -> NDArray[np.int32]:
        return self.frame_rate_per_second * self.period_in_seconds

    @property
    def stopped_image(self) -> Image.Image:
        max_iter = 128
        self.iter = (self.iter + 1) % max_iter
        brightened = self.brighten(self.static_stopped_image, abs((max_iter / 2) - self.iter) / (max_iter / 2))
        noisy = self.salt_and_pepper(brightened, prob=0.1)
        return noisy

    @property
    def stopped_buf(self):
        buf = BytesIO()
        self.stopped_image.save(buf, "JPEG")
        return buf.getbuffer()

    def brighten(self, image: Image.Image, value: float = 1.0):
        brightner = ImageEnhance.Color(image)
        return brightner.enhance(value)

    def salt_and_pepper(self, image: Image.Image, prob: float = 0.05):
        # If the specified `prob` is negative or zero, we don't need to do anything.
        if prob <= 0:
            return image

        arr = np.asarray(image)
        original_dtype = arr.dtype

        # Derive the number of intensity levels from the array datatype.
        intensity_levels: int = 2 ** (arr[0, 0].nbytes * 8)  # pyright: ignore [reportAny]

        min_intensity = 100
        max_intensity = intensity_levels - 1

        # Generate an array with the same shape as the image's:
        # Each entry will have:
        # 1 with probability: 1 - prob
        # 0 or np.nan (50% each) with probability: prob
        random_image_arr = np.random.choice(
            [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
        )

        # This results in an image array with the following properties:
        # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
        # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
        # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
        # We need to to `arr.astype(np.float)` to make sure np.nan is a valid value.
        salt_and_peppered_arr = arr.astype(np.float64) * random_image_arr

        # Since we want SALT instead of NaN, we replace it.
        # We cast the array back to its original dtype so we can pass it to PIL.
        salt_and_peppered_arr = np.nan_to_num(salt_and_peppered_arr, nan=max_intensity).astype(original_dtype)

        return Image.fromarray(salt_and_peppered_arr)


STATIC_CONFIG = StaticConfig()
