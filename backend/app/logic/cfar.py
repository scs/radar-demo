import numpy as np
from numpy.typing import NDArray

from app.logic.model import Model
from app.logic.state import GlobalState


def draw_cross(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    height: int,
    width: int,
    mask_height: int,
    mask_width: int,
    weight: int,
    shape: tuple[int, ...],
):
    if weight % 2 == 0:
        weight = max(weight - 1, 0)

    bottom: int = int(max(0, coord[0] - height))
    bottom_box: int = int(max(0, coord[0] - mask_height))
    top: int = int(min(shape[0], coord[0] + height))
    top_box: int = int(min(shape[0], coord[0] + mask_height))

    left: int = int(min(shape[1], coord[1] + width))
    left_box: int = int(min(shape[1], coord[1] + mask_width))
    right: int = int(max(0, coord[1] - width))
    right_box: int = int(max(0, coord[1] - mask_width))

    width_bottom_: int = int(max(0, coord[0] - weight))
    width_top_: int = int(min(shape[0], coord[0] + weight))
    width_left: int = int(max(0, coord[1] - weight))
    width_right: int = int(min(shape[1], coord[1] + weight))

    for rgb in range(0, 3):
        rgb_image[bottom:bottom_box, width_left:width_right, rgb] = color[rgb]
        rgb_image[top_box:top, width_left:width_right, rgb] = color[rgb]

        rgb_image[width_bottom_:width_top_, right:right_box, rgb] = color[rgb]
        rgb_image[width_bottom_:width_top_, left_box:left, rgb] = color[rgb]


def draw_box(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    height: int,
    width: int,
    weight: int,
    shape: tuple[int, ...],
):
    for i in range(0, weight):
        _draw_box(rgb_image, color, coord, height + i, width + i, shape)


def _draw_box(
    rgb_image: NDArray[np.uint8],
    color: tuple[np.uint8, np.uint8, np.uint8],
    coord: tuple[np.intp, ...],
    height: int,
    width: int,
    shape: tuple[int, ...],
):
    bottom: int = int(max(0, coord[0] - height))
    top: int = int(min(shape[0] - 1, coord[0] + height))
    right: int = int(min(shape[1] - 1, coord[1] + width))
    left: int = int(max(0, coord[1] - width))
    for rgb in range(0, 3):
        rgb_image[bottom, left:right, rgb] = color[rgb]
        rgb_image[top, left:right, rgb] = color[rgb]
        rgb_image[bottom:top, right, rgb] = color[rgb]
        rgb_image[bottom:top, left, rgb] = color[rgb]


def cfar(rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if GlobalState.cfar_enabled():
        coord = np.argmax(rgb_image[..., 0])
        shape_coord: tuple[np.intp, ...] = np.unravel_index(coord, rgb_image[..., 0].shape)
        red: tuple[np.uint8, np.uint8, np.uint8] = (np.uint8(255), np.uint8(0), np.uint8(0))
        stretch: int = 1 if GlobalState.use_hw() else 2
        if GlobalState.model == Model.QUAD_CORNER:
            width = 5 * stretch
            height = 10
            weight = 1
        else:
            width = 3 * stretch
            height = 6
            weight = 2
        shape = rgb_image[..., 0].shape
        draw_box(rgb_image, red, shape_coord, height, width, weight, shape)
        draw_cross(rgb_image, red, shape_coord, 3 * height, 3 * width, height, width, weight, shape)
    return rgb_image
