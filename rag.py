class Rag:
    def __init__(self, retriever_model, model, tokenizer):
        self.retriever_model = retriever_model
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def create_prompt(embedding, query):
        return f"""INSTRUCTION: You are a helpful AI who will ask questions and evaluate the answer from the document.
Embedding: {embedding}
Question: {query}

Your response:
"""

    def rag(self, query, vectorstore, embedding_util):
        query_embedding = self.retriever_model.encode(query)
        retrieved_docs = embedding_util.retrieve(query_embedding, vectorstore)
        context = " ".join(retrieved_docs)
        prompt = self.create_prompt(context, query)
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(inputs)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
