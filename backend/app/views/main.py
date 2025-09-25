import threading
import time
from typing import Any

import psutil
from flask import Response, jsonify, render_template, request

from app import app
from app.logic.benchmark import gen_frames as gen_benchmark_frames
from app.logic.benchmark import start_threads as start_benchmark_threads
from app.logic.benchmark import stop_threads as stop_benchmark_threads
from app.logic.logging import LogLevel, get_logger
from app.logic.model import Model
from app.logic.radar_simulation import gen_frames as gen_radar_frames
from app.logic.radar_simulation import start_threads as start_radar_threads
from app.logic.radar_simulation import stop_threads as stop_radar_threads
from app.logic.state import GlobalState
from app.logic.status import gen_radar_data

#######################################################################################################################
# Module Global Variables
#
HW_LOCK: threading.Lock = threading.Lock()
MINIMAL_UPTIME: int = 20  # seconds

logger = get_logger(__name__, LogLevel.WARNING)


@app.route("/stop")
def stop() -> Response:
    GlobalState.set_stopping()
    return jsonify(GlobalState.get_current_running_state().value)


@app.route("/start")
def start() -> Response:
    GlobalState.set_running()
    return jsonify(GlobalState.get_current_running_state().value)


@app.route("/frame_number")
def frame_number() -> Response:
    r = Response(GlobalState.gen_frame_number_response(), mimetype="text/event-stream")
    return r


@app.route("/video_feed/<int:idx>")
def video_feed(idx: int) -> Response:
    r = Response(gen_radar_frames(idx), mimetype="multipart/x-mixed-replace; boundary=frame")
    return r


@app.route("/imaging_feed")
def imaging_feed() -> Response:
    r = Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")
    return r


@app.route("/short_range_feed")
def short_range_feed() -> Response:
    r = Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")
    return r


@app.route("/benchmark_feed")
def benchmark_feed() -> Response:
    r = Response(gen_benchmark_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return r


@app.route("/radar_data", methods=["GET"])
def radar_data():
    info = gen_radar_data()
    return info


@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify(GlobalState.get_current_settings().to_dict())


@app.route("/settings", methods=["POST"])
def post_settings():
    data: dict[str, Any] = request.get_json()  # pyright: ignore [reportExplicitAny, reportAny]
    id: int = data["id"]  # pyright: ignore [reportAny]
    selectedSetting: dict[str, str] = data["selectedSetting"]  # pyright: ignore [reportAny]
    GlobalState.update_settings(id, selectedSetting)
    return jsonify(GlobalState.get_current_settings().to_dict())


def stop_threads() -> None:
    stop_radar_threads()
    stop_benchmark_threads()


def start_threads() -> None:
    if GlobalState.model == Model.ONE_D_FFT:
        start_benchmark_threads()
    elif GlobalState.model in [Model.SHORT_RANGE, Model.QUAD_CORNER, Model.IMAGING]:
        start_radar_threads()


@app.route("/initNewModel", methods=["POST"])
def init_new_model():

    if HW_LOCK.acquire(blocking=True):
        stop_threads()
        data: dict[str, Any] = request.get_json()  # pyright: ignore [reportExplicitAny, reportAny]
        model: str = data["demoModel"]  # pyright: ignore [reportAny]
        GlobalState.init_state(model)
        start_threads()

    HW_LOCK.release()

    return jsonify(GlobalState.get_current_state())


@app.route("/leavePage", methods=["Get"])
def leave_page():
    # this function is now obosolete
    return "", 200


@app.route("/initApp", methods=["GET"])
def init_app():
    uptime = time.time() - psutil.boot_time()
    if uptime < MINIMAL_UPTIME:
        time.sleep(MINIMAL_UPTIME - uptime)

    GlobalState.init_state(None)
    return "", 200


@app.route("/")
def index():
    return render_template("index.html")
