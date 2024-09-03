import faiss
import numpy as np
from pymongo import MongoClient

class VectorSearch:
    def __init__(self, db_name, collection_name, uri="mongodb://localhost:27017/"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
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
        Load vectors from the database.
        """
        vectors = []
        self.ids = []

        for document in self.collection.find():
            vector = np.array(document["vector"], dtype=np.float32)
            vectors.append(vector)
            self.ids.append(document["_id"])

        return np.array(vectors)

    def add_vector(self, new_vector):
        """
        Add a new vector to the index and save it.
        """
        new_vector = np.array(new_vector, dtype=np.float32).reshape(1, -1)
        self.index.add(new_vector)
        faiss.write_index(self.index, self.index_file)

    def search(self, query_vector, k=5):
        """
        Search the index for similar vectors.
        """
        query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            results.append(self.collection.find_one({"_id": self.ids[idx]}))
        
        return results

    def close_connection(self):
        """
        Close the MongoDB connection.
        """
        self.client.close()
