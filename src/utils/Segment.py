import numpy as np
from bson.binary import Binary


class Segment:
    def __init__(self, vector, path):
        """
        Initialize a Segment object with a 128-float vector and a 2D numpy array.
        """
        if len(vector) != 128:
            raise ValueError("Vector must be 128 floats long.")
        self.vector = vector
        self.bson_path = self.encode_path(path)

    def encode_path(self, path):
        """
        Encode a numpy 2D array into BSON format.
        """
        # Convert the numpy array to a bytes object
        np_bytes = np.ndarray.tobytes(path)
        # Encode the bytes object to BSON Binary format
        return Binary(np_bytes)

    def get_path(self):
        """
        Gets the path as a numpy array.
        """
        # Decode the BSON Binary data to a bytes object
        np_bytes = bytes(self.bson_path)
        # Convert the bytes object back to a numpy array
        path = np.frombuffer(np_bytes, dtype=np.float64)
        # Reshape the array as needed (you need to know the shape in advance)
        return path.reshape(-1, 2)  # Example shape, adjust as necessary

    def to_dict(self):
        """
        Convert the Segment object to a dictionary for MongoDB storage.
        """
        return {
            "vector": self.vector,
            "bson_data": self.bson_path
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Segment object from a dictionary retrieved from MongoDB.
        """
        path = cls.get_path(cls, data["bson_path"])
        return cls(vector=data["vector"], path=path)