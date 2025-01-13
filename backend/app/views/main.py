import logging
import threading
from time import sleep, time
from typing import Any

from flask import Response, jsonify, render_template, request

from app import app
from app.logic.benchmark import gen_frames as gen_benchmark_frames
from app.logic.radar_simulation import gen_frames as gen_radar_frames
from app.logic.state import GlobalState
from app.logic.status import gen_radar_data

#######################################################################################################################
# Module Global Variables
#
log_level = logging.ERROR  # NOTSET, DEBUG, INFO, WARNING, ERROR
LEAVE_PAGE_LOCK: threading.Lock = threading.Lock()


class CustomFormatter(logging.Formatter):

    grey: str = "\x1b[38;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    format_str: str = "%(levelname)-8s (%(filename)-26s:%(lineno)3d:%(funcName)-30s) - %(message)s "

    FORMATS: dict[int, str] = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record: logging.LogRecord):  # pyright: ignore [reportImplicitOverride]
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger_stream = logging.StreamHandler()
logger_stream.setLevel(log_level)
# formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s")
logger_stream.setFormatter(CustomFormatter())
logger.setLevel(log_level)
logger.addHandler(logger_stream)
logger.propagate = False


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
    return "", 200


@app.route("/initApp", methods=["GET"])
def init_app():
    GlobalState.init_state(None)
    return "", 200


@app.route("/")
def index():
    return render_template("index.html")
