import logging
from app import app

if __name__ == "__main__":
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
