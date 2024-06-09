import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Vectorstore:
    def __init__(self, collection_name, host='localhost', port=8000):
        self.collection_name = collection_name
        self.client = chromadb.Client(host=host, port=port)
        self.chroma_collection = self.client.create_collection(collection_name)

    def add(self, documents):
        self.chroma_collection.add_documents(documents)

    def retrieve(self, query_embedding, top_k=5):
        all_documents = self.chroma_collection.get_all_documents()
        embeddings = [doc['embedding'] for doc in all_documents]
        similarities = cosine_similarity([query_embedding], embeddings).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [all_documents[i]['content'] for i in top_indices]
