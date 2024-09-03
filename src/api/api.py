from flask import Flask, jsonify, request
from flask_httpauth import HTTPTokenAuth
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from ..data_processing import DataUpdater
from ..services import VectorSearch, MongoDBHandler

class APIService:
    def __init__(self, app: Flask):
        self.app = app
        self.jwt = JWTManager(app)
        self.auth = HTTPTokenAuth(scheme='Bearer')

        # Dummy user data for the example
        self.fake_users_db = {
            "admin": {"username": "admin", "full_name": "Admin User", "password": "admin", "role": "admin"},
            "user": {"username": "user", "full_name": "Normal User", "password": "user", "role": "user"},
        }

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        @self.auth.verify_token
        def verify_token(token):
            try:
                user = get_jwt_identity()
                if user in self.fake_users_db:
                    return self.fake_users_db[user]
            except:
                return None

        @self.app.route('/token', methods=['POST'])
        def login():
            return self._login()

        @self.app.route('/admin/update_data', methods=['POST'])
        @jwt_required()
        def update_data():
            return self._update_data()

        @self.app.route('/search', methods=['GET'])
        def search_data():
            return self._search_data()

    def _login(self):
        username = request.form.get('username')
        password = request.form.get('password')
        user = self.fake_users_db.get(username)
        if user and user['password'] == password:
            access_token = create_access_token(identity=username)
            return jsonify(access_token=access_token)
        return jsonify({"message": "Invalid credentials"}), 401

    def _update_data(self):
        current_user = get_jwt_identity()
        if self.fake_users_db[current_user]['role'] != 'admin':
            return jsonify({"message": "Admin privileges required"}), 403

        data = request.get_json()
        image_name = data.get('image_name')

        if not image_name:
            return jsonify({"message": "Image name is required"}), 400

        # Create an instance of DataUpdater and update the database
        updater = DataUpdater(image_name)
        updater.update_database()

        return jsonify({"message": f"Database updated successfully for image: {image_name}"})

    def _search_data(self):
        # Get the query from the request
        query_str = request.args.get('query', '')

        try:
            # Convert the query string into a list of floats
            query = [float(num) for num in query_str.strip('[]').split(',')]
        except ValueError:
            return jsonify({"error": "Invalid input format"}), 400

        # Perform the search with the array of floats
        print(query)
        dbHandler = MongoDBHandler(db_name="your_db_name", collection_name="your_collection_name")
        vectorSearch = VectorSearch(dbHandler=dbHandler)
        vectorSearch.load_or_build_index()
        result = vectorSearch.search(query)
        
        return jsonify(result)
