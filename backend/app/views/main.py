import logging
import threading
from time import sleep, time
from typing import Any

from flask import Response, jsonify, render_template, request

from app import app
from app.logic.benchmark import gen_frames as gen_benchmark_frames
from app.logic.logging import LogLevel, get_logger
from app.logic.radar_simulation import gen_frames as gen_radar_frames
from app.logic.state import GlobalState
from app.logic.status import gen_radar_data

#######################################################################################################################
# Module Global Variables
#
LEAVE_PAGE_LOCK: threading.Lock = threading.Lock()
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
    return Response(GlobalState.gen_frame_number_response(), mimetype="text/event-stream")


@app.route("/video_feed/<int:idx>")
def video_feed(idx: int) -> Response:
    return Response(gen_radar_frames(idx), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/imaging_feed")
def imaging_feed() -> Response:
    return Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/short_range_feed")
def short_range_feed() -> Response:
    return Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/benchmark_feed")
def benchmark_feed() -> Response:
    return Response(gen_benchmark_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/radar_data", methods=["GET"])
def radar_data():
    info = gen_radar_data()
    return info


@app.route("/settings", methods=["GET"])
def get_settings():
    return jsonify(GlobalState.get_current_settings().to_dict())


@app.route("/settings", methods=["POST"])
def post_settings():
    data: dict[str, Any] = request.get_json()  # pyright: ignore [reportExplicitAny]
    id: int = data["id"]
    selectedSetting: dict[str, str] = data["selectedSetting"]
    GlobalState.update_settings(id, selectedSetting)
    return jsonify(GlobalState.get_current_settings().to_dict())


@app.route("/initNewModel", methods=["POST"])
def init_new_model():
    data: dict[str, Any] = request.get_json()  # pyright: ignore [reportExplicitAny]
    model: str = data["demoModel"]
    GlobalState.init_state(model)
    return jsonify(GlobalState.get_current_state())


@app.route("/leavePage", methods=["Get"])
def leave_page():
    logger.info("Entering leave_page")

    def timeout(start: float) -> bool:
        TIMEOUT = 1  # second(s)
        return time() - start > TIMEOUT

    _ = LEAVE_PAGE_LOCK.acquire()
    if GlobalState.mutex_lock.locked():
        GlobalState.stop_producer.set()
        start = time()
        while GlobalState.stop_producer.is_set():
            sleep(0.01)
            if timeout(start):  # [seconds]
                logger.error("Unable to stop producer")
                break

    LEAVE_PAGE_LOCK.release()
    logger.info("Leaving leave_page")
    return "", 200


@app.route("/initApp", methods=["GET"])
def init_app():
    GlobalState.init_state(None)
    return "", 200


@app.route("/")
def index():
    return render_template("index.html")
