from enum import Enum


class Model(Enum):
    ONE_D_FFT = "ONE_D_FFT"
    SHORT_RANGE = "SHORT_RANGE"
    QUAD_CORNER = "QUAD_CORNER"
    IMAGING = "IMAGING"
    NONE = "NONE"


MODEL_LOOKUP = {"NONE": 0, "ONE_D_FFT": 1, "SHORT_RANGE": 2, "QUAD_CORNER": 3, "IMAGING": 4}
