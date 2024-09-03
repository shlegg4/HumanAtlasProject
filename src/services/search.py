import faiss
import numpy as np
from pymongo import MongoClient
from .database import MongoDBHandler  # Adjust this import based on your module structure


class VectorSearch:
    def __init__(self, dbHandler:MongoDBHandler):
        self.mongodb_handler = dbHandler
        self.index = None
        self.index_file = "faiss_index.idx"
        faiss.omp_set_num_threads(4)  # Enable multi-threading with 4 threads

    def load_or_build_index(self, d=128):
        """
        Load the index from disk, or build it if it doesn't exist.
        """
        try:
            self.index = faiss.read_index(self.index_file)
            print("Index loaded from disk.")
        except:
            print("Building a new index.")
            vectors = self.get_vectors()
            self.index = faiss.IndexFlatL2(d)
            self.index.add(vectors)
            faiss.write_index(self.index, self.index_file)

    def get_vectors(self):
        """
        Load vectors from the database using MongoDBHandler.
        """
        segments = self.mongodb_handler.collection.find()
        vectors = []
        self.ids = []

        for segment in segments:
            vector = np.array(segment["vector"], dtype=np.float32)
            vectors.append(vector)
            self.ids.append(segment["_id"])

        return np.array(vectors)

    def add_vector(self, new_vector, segment=None):
        """
        Add a new vector to the index, save it, and store it in the database using MongoDBHandler.
        """
        new_vector = np.array(new_vector, dtype=np.float32).reshape(1, -1)
        self.index.add(new_vector)
        faiss.write_index(self.index, self.index_file)

        if segment:
            segment.vector = new_vector.flatten().tolist()
            self.mongodb_handler.insert_segment(segment)

    def search(self, query_vector, k=5):
        """
        Search the index for similar vectors.
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            result = self.mongodb_handler.find_by_id(self.ids[idx])
            results.append(result)
        
        return results

    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        self.mongodb_handler.close_connection()


# Example usage:
# vector_search = VectorSearch(db_name="your_db", collection_name="your_collection")
# vector_search.load_or_build_index(d=128)
# query_vector = [0.1, 0.2, ...]  # Your query vector here
# results = vector_search.search(query_vector)
# vector_search.close_connection()
