from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
_ = CORS(app)

from app.views import main
