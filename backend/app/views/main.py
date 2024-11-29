from typing import Any

from flask import Response, jsonify, render_template, request

from app import app
from app.logic.benchmark import gen_frames as gen_benchmark_frames
from app.logic.radar_simulation import gen_frames as gen_radar_frames
from app.logic.state import GlobalState
from app.logic.status import gen_radar_data


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
    data: dict[str, Any] = request.get_json()
    id: int = data["id"]
    selectedSetting: dict[str, str] = data["selectedSetting"]
    GlobalState.update_settings(id, selectedSetting)
    return jsonify(GlobalState.get_current_settings().to_dict())


@app.route("/initNewModel", methods=["POST"])
def init_new_model():
    data: dict[str, Any] = request.get_json()
    model: str = data["demoModel"]
    GlobalState.init_state(model)
    return jsonify(GlobalState.get_current_state())


@app.route("/")
def index():
    return render_template("index.html")