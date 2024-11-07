from app import app
from app.logic.state import GlobalState

if __name__ == "__main__":
    GlobalState.init_state(None)
    app.run(debug=False, threaded=True)
