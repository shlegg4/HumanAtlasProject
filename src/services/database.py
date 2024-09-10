from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections
)
from ..utils import Segment, log_message

class MilvusHandler:
    def __init__(self, collection_name, host="milvus-standalone", port="19530"):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.connect()
        self.create_collection()

    def connect(self):
        if not connections.has_connection("default"):
            connections.connect("default", host=self.host, port=self.port)
            log_message('info', 'Connected to Milvus.')
        else:
            log_message('info', 'Already connected to Milvus.')
            
    def create_collection(self):
        """
        Create a collection in Milvus with a vector field (128 dimensions) and a BSON string field.
        """
        # Define the fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto-incrementing ID
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),  # 128-dimensional vector
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),  # BSON data stored as VARCHAR
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=256)
        ]

        schema = CollectionSchema(fields, description="Collection for vectors and BSON data")
        
        # Create the collection (if not exists)
        if self.collection_name not in list_collections():
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.create_index()
            log_message('info', f'Collection {self.collection_name} created.')
        else:
            self.collection = Collection(self.collection_name)
            log_message('info', f'Collection {self.collection_name} already exists.')
        
        
        self.collection.load()

    def create_index(self, index_type="IVF_FLAT", metric_type="L2", nlist=128):
        """
        Create an index on the vector field to allow for efficient vector search.
        
        Parameters:
        index_type: The type of index (e.g., IVF_FLAT, IVF_SQ8, etc.)
        metric_type: The distance metric (e.g., L2, IP)
        nlist: Number of clusters used for index (affects search speed and accuracy)
        """
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,  # L2 distance by default
            "params": {"nlist": nlist}   # Number of clusters (affects search performance)
        }
        # Create index on the "vector" field
        self.collection.create_index("vector", index_params)
        log_message('info', f'Index {index_type} created on the vector field.')

    def insert_segment(self, segment):
        """
        Insert a Segment object into the Milvus collection.
        """
        vector = segment.vector  # The vector (128-dim float list)
        path = segment.path  # The BSON data stored as a string
        url = segment.url
        # Insert data into Milvus
        data = [
            [vector],  # Vectors should be in a nested list (even if inserting one vector)
            [path],
            [url]
        ]
        result = self.collection.insert(data)
        log_message('info', f'Segment with vector inserted into Milvus.')
        return result.primary_keys

    def get_segments(self):
        """
        Retrieve all segments in the collection (returning as Segment objects).
        """
        # Query all vectors and BSON fields
        results = self.collection.query(expr="id >= 0", output_fields=["vector", "path"])
        segments = []
        for result in results:
            segment = Segment.from_dict({"vector": result['vector'], "path": result['path']})
            segments.append(segment)
        
        return segments

    def find_by_vector(self, vector, top_k=1):
        """
        Search for a segment by vector similarity (using L2 distance by default).
        """
        # Search for top_k most similar vectors
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search([vector], "vector", param=search_params, limit=top_k, output_fields=["vector", "path", "url"])
        
        if results:
            # Assuming the search returns BSON fields with the vector
            log_message('info', f'results {results}')
            segment_dict = {"vector":results[0][0].entity.get("vector"), "path":results[0][0].entity.get("path"), "url":results[0][0].entity.get("url")}
            segment = Segment.from_dict(segment_dict)
            return segment
        return None

    def find_by_id(self, segment_id):
        """
        Find a Segment by its _id (Milvus's auto-incrementing ID).
        """
        result = self.collection.query(expr=f"id == {segment_id}", output_fields=["vector", "path"])
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
            "path": new_segment.path
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
