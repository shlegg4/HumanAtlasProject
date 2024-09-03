from pymongo import MongoClient
from ..utils import Segment

class MongoDBHandler:
    def __init__(self, db_name, collection_name, uri="mongodb://localhost:27017/"):
        """
        Initialize the MongoDBHandler with the database and collection names.
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def insert_segment(self, segment):
        """
        Insert a Segment object into the collection.
        """
        document = segment.to_dict()
        result = self.collection.insert_one(document)
        return result.inserted_id

    def find_by_vector(self, vector):
        """
        Find a Segment by the vector.
        """
        data = self.collection.find_one({"vector": vector})
        return Segment.from_dict(data) if data else None

    def update_segment(self, vector, new_segment=None):
        """
        Update a Segment's vector and/or BSON data based on the original vector.
        """
        update_fields = {}
        if new_segment:
            update_fields["vector"] = new_segment.vector
            update_fields["bson_path"] = new_segment.bson_data

        if not update_fields:
            raise ValueError("No fields to update provided.")

        result = self.collection.update_one(
            {"vector": vector},
            {"$set": update_fields}
        )
        return result.modified_count

    def delete_by_vector(self, vector):
        """
        Delete a Segment based on the vector.
        """
        result = self.collection.delete_one({"vector": vector})
        return result.deleted_count

    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()