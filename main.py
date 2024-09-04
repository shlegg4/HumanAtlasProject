from flask import Flask
from src.api.api import APIService
from src.utils import setup_logger

def create_app():
    app = Flask(__name__)
    app.config['JWT_SECRET_KEY'] = 'your_jwt_secret_key'  # Change this to a random secret key

    # Setup logging util
    setup_logger()

    # Initialize API service
    api_service = APIService(app)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
