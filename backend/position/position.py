import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin


def sqsin(phase: float) -> float:
    return sin(phase) ** 2 + 1


def compute_position(
    phase: float,
    amplitude: tuple[float, float, float] = (1, 1, 1),
    offset: tuple[float, float, float] = (0, 0, 0),
) -> dict[str, float]:
    return {
        "x": float((sin(2 * phase)) / sqsin(phase) * amplitude[0] + offset[0]),
        "y": float((1 - cos(phase) / sqsin(phase)) * amplitude[1] + offset[1]),
        "z": float(0 * amplitude[2] + offset[2]),
    }


def compute_velocity_difference(
    phase: float,
    delta: float,
    amplitude: tuple[float, float, float] = (1, 1, 1),
) -> dict[str, float]:
    pos0 = compute_position(phase, amplitude)
    pos1 = compute_position(phase + delta, amplitude)
    return {
        "x": (pos1["x"] - pos0["x"]) / delta,
        "y": (pos1["y"] - pos0["y"]) / delta,
        "z": (pos1["z"] - pos0["z"]) / delta,
    }


def compute_velocity_derivate(
    phase: float,
    amplitude: tuple[float, float, float] = (1, 1, 1),
) -> dict[str, float]:
    return {
        "x": (2 * cos(2 * phase) * sqsin(phase) - sin(2 * phase) ** 2) / sqsin(phase) * amplitude[0],
        "y": (-(sin(phase) ** 3) - sin(phase) - sin(2 * phase) * cos(phase)) / sqsin(phase) ** 2 * amplitude[1],
        "z": 0 * amplitude[2],
    }


if __name__ == "__main__":
    waves = np.empty((40, 3), dtype=float)
    for step in range(40):
        d = compute_position(2 * np.pi * step / 40)
        waves[step, :] = [d["x"], d["y"], d["z"]]
    plt.figure()
    plt.plot(waves)
    plt.show()
