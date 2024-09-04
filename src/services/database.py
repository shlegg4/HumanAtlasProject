from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection
)
from ..utils import Segment, log_message

class MilvusHandler:
    def __init__(self, collection_name, host="localhost", port="19530"):
        """
        Initialize the MilvusHandler with the collection name and connect to Milvus.
        """
        connections.connect("default", host=host, port=port)
        self.collection_name = collection_name
        self.create_collection()

    def create_collection(self):
        """
        Create a collection in Milvus with a vector field (128 dimensions) and a BSON string field.
        """
        # Define the fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-incrementing ID
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),  # 128-dimensional vector
            FieldSchema(name="bson_field", dtype=DataType.VARCHAR, max_length=1024)  # BSON data stored as VARCHAR
        ]

        schema = CollectionSchema(fields, description="Collection for vectors and BSON data")
        
        # Create the collection (if not exists)
        if self.collection_name not in Collection.list():
            self.collection = Collection(name=self.collection_name, schema=schema)
            log_message('info', f'Collection {self.collection_name} created.')
        else:
            self.collection = Collection(self.collection_name)
            log_message('info', f'Collection {self.collection_name} already exists.')

    def insert_segment(self, segment):
        """
        Insert a Segment object into the Milvus collection.
        """
        vector = segment.vector  # The vector (128-dim float list)
        bson_data = segment.bson_data  # The BSON data stored as a string

        # Insert data into Milvus
        data = [
            [vector],  # Vectors should be in a nested list (even if inserting one vector)
            [bson_data]
        ]
        result = self.collection.insert(data)
        log_message('info', f'Segment with vector inserted into Milvus.')
        return result.insert_ids

    def get_segments(self):
        """
        Retrieve all segments in the collection (returning as Segment objects).
        """
        # Query all vectors and BSON fields
        results = self.collection.query(expr="id >= 0", output_fields=["vector", "bson_field"])
        segments = []
        for result in results:
            segment = Segment.from_dict(result)  # Assuming from_dict can handle vector and bson
            segments.append(segment)
        
        return segments

    def find_by_vector(self, vector, top_k=1):
        """
        Search for a segment by vector similarity (using L2 distance by default).
        """
        # Search for top_k most similar vectors
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search([vector], "vector", param=search_params, limit=top_k, output_fields=["bson_field"])

        if results:
            # Assuming the search returns BSON fields with the vector
            segment = Segment.from_dict({"vector": vector, "bson_field": results[0].entity.bson_field})
            return segment
        return None

    def find_by_id(self, segment_id):
        """
        Find a Segment by its _id (Milvus's auto-incrementing ID).
        """
        result = self.collection.query(expr=f"id == {segment_id}", output_fields=["vector", "bson_field"])
        if result:
            return Segment.from_dict(result[0])  # Assuming from_dict handles the dict format
        return None

    def update_segment(self, segment_id, new_segment=None):
        """
        Update a segment based on its ID.
        """
        if not new_segment:
            raise ValueError("No fields to update provided.")

        update_data = {
            "vector": new_segment.vector,
            "bson_field": new_segment.bson_data
        }

        result = self.collection.update([segment_id], update_data)
        log_message('info', f'Segment with ID {segment_id} updated.')
        return result

    def delete_by_vector(self, vector):
        """
        Delete a segment by its vector.
        """
        # Milvus doesn’t support deletion by vector directly, so we need to first search for the vector
        search_result = self.find_by_vector(vector)
        if search_result:
            result = self.collection.delete(expr=f"id == {search_result.id}")
            return result
        return 0  # No vector found

    def close_connection(self):
        """
        Close the connection to Milvus.
        """
        connections.disconnect("default")
