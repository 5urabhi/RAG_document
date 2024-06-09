from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def load_document(self, pdf_path):
        text = pdf2text.convert(pdf_path)
        return text

    def create_embeddings(self, text_chunks):
        embeddings = []
        for chunk in text_chunks:
            embedding = self.model.encode(chunk)
            embeddings.append(embedding)
        return embeddings
