from embedding import Embedding
from vector import Vectorstore
from model import Model
from rag import Rag
import pdf2text

# Initialize Embedding utility with model name
embedding_util = Embedding(model_name='all-MiniLM-L6-v2')

vectorstore = Vectorstore("my_collection", host='localhost', port=8000)


# Initialize Model
model_util = Model()
model = model_util.load_model()
tokenizer = model_util.load_tokenizer()

# Load and process PDF
file_path = 'document2.pdf'
text = embedding_util.load_document(file_path)
text_chunks = text.split('. ')

# Create embeddings
embeddings = embedding_util.create_embeddings(text_chunks)

# Add documents to Vectorstore
documents = [{"content": chunk, "embedding": embedding, "metadata": {"source": "pdf_document", "page_no": i}} for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings))]
vectorstore.add(documents)

# Initialize RAG
retriever_model = embedding_util.model
rag_util = Rag(retriever_model, model, tokenizer)

# Example question
question = "What is the main topic of the document?"

# Get answer using RAG
answer = rag_util.rag(question, vectorstore, embedding_util)
print(answer)
