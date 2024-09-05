from flask import Flask, jsonify, request
from flask_httpauth import HTTPTokenAuth
from flask_cors import CORS
from ..data_processing import DataUpdatePipeline, DataSearchPipeline
from ..utils import log_message
from skimage import io

class APIService:
    def __init__(self, app: Flask):
        self.app = app
        self.auth = HTTPTokenAuth(scheme='Bearer')
        CORS(self.app)
        # Dummy user data for the example
        self.fake_users_db = {
            "admin": {"username": "admin", "full_name": "Admin User", "password": "admin", "role": "admin"},
            "user": {"username": "user", "full_name": "Normal User", "password": "user", "role": "user"},
        }

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/token', methods=['POST'])
        def login():
            return self._login()

        @self.app.route('/admin/update_data', methods=['POST'])
        def update_data():
            return self._update_data()

        @self.app.route('/search', methods=['GET'])
        def search_data():
            return self._search_data()

    def _login(self):
        username = request.form.get('username')
        password = request.form.get('password')
        user = self.fake_users_db.get(username)

        log_message('info', f'Generating token for {user}')

        log_message('warning', f'{user} attempted logging in with invalid credentials.')
        return jsonify({"message": "Invalid credentials"}), 401

    def _update_data(self):

        data = request.get_json()
        image_name = data.get('image_name')
        log_message('info', f'user is adding {image_name} to database')

        if not image_name:
            return jsonify({"message": "Image name is required"}), 400

        # Create an instance of DataUpdater and update the database
        updater = DataUpdatePipeline(image_name)
        updater.update_database()

        log_message('info', f'Successfully added image to database')
        return jsonify({"message": f"Database updated successfully for image: {image_name}"})

    def _search_data(self):
        # Get the query from the request
        query_str = request.args.get('query', '')
        log_message('info', f'searching data with image {query_str}')
        
        image = io.imread(query_str)
        searcher = DataSearchPipeline()
        result = searcher.search(image=image)
        return jsonify(result)
