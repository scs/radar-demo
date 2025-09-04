import ctypes
from typing import final

# Define ctypes structures for the headers and their payloads


@final
class Header(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int32),
        ("minus1", ctypes.c_int32),
        ("reserved", ctypes.c_int32 * 6),
    ]


@final
class DopplerRangeEntry(ctypes.Structure):
    _fields_ = [
        ("doppler", ctypes.c_int16),
        ("range", ctypes.c_int16),
    ]


@final
class AoAEntry(ctypes.Structure):
    _fields_ = [
        ("azimuth", ctypes.c_float),
        ("elevation", ctypes.c_float),
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("velocity", ctypes.c_float),
        ("reserved", ctypes.c_float * 2),
    ]


@final
class DataBlob(ctypes.Structure):
    _fields_ = [
        ("range_doppler_data", ctypes.c_int16 * (1024 * 1024 // 2)),
        ("cfar_header", Header),
        ("cfar_payload", DopplerRangeEntry * 262136),
        ("aoa_header", Header),
        ("aoa_payload", AoAEntry * 32767),
    ]


if __name__ == "__main__":
    data_blob = DataBlob()

    # Initialize all data in DataBlob to zero
    ctypes.memset(ctypes.addressof(data_blob), 0, ctypes.sizeof(data_blob))

    print(f"Size of DataBlob: {ctypes.sizeof(data_blob)} bytes")
