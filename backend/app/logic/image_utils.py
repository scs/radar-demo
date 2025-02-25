from io import BytesIO

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.logic.timer import Timer


def convert_to_intensity_image(complex_result: NDArray[np.int16]) -> NDArray[np.int32]:
    timer = Timer(name="convert_to_intensity_image")
    inphase = complex_result[..., 0].astype(float)
    quadrature = complex_result[..., 1].astype(float)
    power = np.square(inphase) + np.square(quadrature)
    intensity_images = np.log10(power + 1)
    max: float = np.amax(intensity_images)
    intensity_images = intensity_images / max * 255
    intensity_images = np.roll(np.clip(intensity_images, 0, 255).astype(np.int32), 256, axis=2)
    timer.log_time()
    return intensity_images


def norm_image(image: NDArray[np.int16]) -> NDArray[np.uint8]:
    max: np.int16 = np.amax(image)
    norm_image = image / max * 255
    return norm_image.astype(np.uint8)


def heat_map(intensity_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    timer = Timer(name="heat_map")
    flipped = np.flip(intensity_image, axis=0)
    img = np.empty((*flipped.shape, 3), dtype=np.uint8)
    img[..., 0] = np.where(flipped < 128, 0, (flipped - 128) * 2)  # red
    img[..., 1] = np.where(flipped < 128, flipped * 2, 255)  # green
    img[..., 2] = np.where(flipped < 128, 255 - flipped * 2, 0)  # blue
    timer.log_time()
    return img


def create_frame(rgb_array: NDArray[np.uint8]):
    timer: Timer = Timer(name="create_frame")
    buf: BytesIO = BytesIO()
    image: Image.Image = Image.fromarray(rgb_array, "RGB")
    image.save(buf, "JPEG")
    frame: memoryview[int] = buf.getbuffer()
    timer.log_time()
    return frame
