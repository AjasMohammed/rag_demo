from typing import Iterable
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


class QdrantDB:
    """
    A class to manage interaction with a Qdrant database for text embeddings.
    """

    def __init__(self, collection_name, url="http://localhost:6333"):
        """
        Initialize the QdrantDB with a collection name and URL for the Qdrant client.
        """
        self.client: QdrantClient = QdrantClient(url=url)
        self.collection_name: str = collection_name
        self.embedder: TextEmbedding = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5")

    def embed_texts(self, text: Iterable[str] | str) -> Iterable[list[float]] | list[float]:
        """
        Embed a list of texts or a single text string into vector embeddings.
        """
        return self.embedder.embed(text)

    def create_collections(self) -> bool:
        """
        Create a new collection in Qdrant if it does not already exist.
        """
        if self.client.collection_exists(self.collection_name):
            print('Collection already exists!')
            return False
        print('Creating collection...')
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=384, distance=models.Distance.COSINE,
            ),
        )
        return True

    def delete_collection(self) -> None:
        """
        Delete the specified collection from Qdrant if it exists.
        """
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            print('Collection deleted!')
        else:
            print('Collection does not exist!')

    def create_points(self, embeddings: list[list[float]], payload: list[dict]) -> list[models.PointStruct]:
        """
        Create point structures for embeddings and their corresponding payloads.
        """
        points = [
            models.PointStruct(
                id=payload[i]['id'],
                vector=embeddings[i],
                payload=payload[i]
            )
            for i in range(len(embeddings))
        ]
        return points

    def ingest_data(self, points: list[models.PointStruct]) -> models.UpdateResult:
        """
        Insert or update points in the Qdrant collection.
        """
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return operation_info

    def query_collections(self, query_vector: list[float], limit: int = 3) -> list:
        """
        Search the collection for similar vectors using the query vector.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results


if __name__ == "__main__":
    # Example usage
    text = ["hello world", "hello universe", "hello cosmos"]
    # Initialize QdrantDB and embed texts
