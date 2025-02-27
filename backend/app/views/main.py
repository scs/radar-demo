import threading
from typing import Any

from flask import Response, jsonify, render_template, request

from app import app
from app.logic.benchmark import gen_frames as gen_benchmark_frames
from app.logic.benchmark import start_threads as start_benchmark_threads
from app.logic.benchmark import stop_threads as stop_benchmark_threads
from app.logic.logging import LogLevel, get_logger
from app.logic.radar_simulation import gen_frames as gen_radar_frames
from app.logic.radar_simulation import start_threads as start_radar_threads
from app.logic.radar_simulation import stop_threads as stop_radar_threads
from app.logic.state import GlobalState
from app.logic.status import gen_radar_data

#######################################################################################################################
# Module Global Variables
#
HW_LOCK: threading.Lock = threading.Lock()
QUAD_RADAR_LOCK: list[threading.Lock] = [threading.Lock(), threading.Lock(), threading.Lock(), threading.Lock()]


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


def coc() -> None:
    print("################################################################################")
    print("#                              COC Called                                      #")
    print("################################################################################")


@app.route("/video_feed/<int:idx>")
def video_feed(idx: int) -> Response:
    _ = QUAD_RADAR_LOCK[idx].acquire()
    r = Response(gen_radar_frames(idx), mimetype="multipart/x-mixed-replace; boundary=frame")
    if idx == 0:
        _ = HW_LOCK.acquire()
        start_radar_threads()

    def coc() -> None:
        if idx == 0:
            stop_radar_threads()
            HW_LOCK.release()
        QUAD_RADAR_LOCK[idx].release()
        print("################################################################################")
        print(f"#                              COC Called for idx = {idx}                          #")
        print("################################################################################")

    _ = r.call_on_close(coc)
    return r


@app.route("/imaging_feed")
def imaging_feed() -> Response:
    _ = HW_LOCK.acquire()
    start_radar_threads()
    r = Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")

    def coc() -> None:
        stop_radar_threads()
        print("####################################################")
        print("#              SHORT RANGE COC                      #")
        print("####################################################")
        HW_LOCK.release()

    _ = r.call_on_close(coc)
    return r


@app.route("/short_range_feed")
def short_range_feed() -> Response:
    _ = HW_LOCK.acquire()
    start_radar_threads()
    r = Response(gen_radar_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")

    def coc() -> None:
        stop_radar_threads()
        print("####################################################")
        print("#              SHORT RANGE COC                      #")
        print("####################################################")
        HW_LOCK.release()

    _ = r.call_on_close(coc)
    return r


@app.route("/benchmark_feed")
def benchmark_feed() -> Response:
    HW_LOCK.acquire()
    start_benchmark_threads()
    r = Response(gen_benchmark_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def coc() -> None:
        stop_benchmark_threads()
        print("####################################################")
        print("#              BENCHMARK COC                       #")
        print("####################################################")
        HW_LOCK.release()

    _ = r.call_on_close(coc)
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
    logger.info("Leaving leave_page")
    return "", 200


@app.route("/initApp", methods=["GET"])
def init_app():
    GlobalState.init_state(None)
    return "", 200


@app.route("/")
def index():
    return render_template("index.html")
