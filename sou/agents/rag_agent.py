import torch
import torch.nn.functional as F
from pydantic import BaseModel

from sou.models.embedding_model import EmbeddingModel
from sou.models.generation_model import Model as GenerationModel

from .agent import Agent


class Document(BaseModel):
    content: str
    embedding: list[float]


class RAGAgent(Agent):
    def __init__(
        self,
        llm_model: GenerationModel,
        embedding_model: EmbeddingModel,
        name: str = "rag_agent",
        chunk_size: int = 1024,
        chunk_overlap: int = 50,
    ) -> None:
        self.name = name
        self.documents: list[Document] = []
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.device = embedding_model.device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use the tokenizer from the provided embedding model
        self.tokenizer = self.embedding_model.tokenizer

    def purge_document_base(self):
        self.documents = []

    def chunk_document(self, document: str) -> list[str]:
        tokens = self.tokenizer.tokenize(document)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = self.tokenizer.convert_tokens_to_string(tokens[start:end])
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def store_documents(self, documents: list[str]):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))

        embeddings = self.embedding_model.generate_embeddings(all_chunks)

        documents = [
            Document(content=chunk, embedding=embedding)
            for chunk, embedding in zip(all_chunks, embeddings)
        ]
        self.documents.extend(documents)

    def retrieve_documents(self, query: str) -> list[Document]:
        query_embedding = self.embedding_model.generate_embedding(query)
        scores = []
        for doc in self.documents:
            score = F.cosine_similarity(
                torch.tensor([query_embedding]).to(self.device),
                torch.tensor([doc.embedding]).to(self.device),
            )
            scores.append((doc, score.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:5]]

    def query(self, query: str) -> str:
        retrieved_docs = self.retrieve_documents(query)
        context = "\n".join([doc.content for doc in retrieved_docs])

        messages = [
            {
                "role": "system",
                "content": PROMPT.format(question=query, context=context),
            },
        ]
        response = self.llm_model.generate_response(messages)

        return response

    def summary_context(self, query: str) -> str:
        retrieved_docs = self.retrieve_documents(query)
        context = "\n".join([doc.content for doc in retrieved_docs])

        messages = [
            {
                "role": "system",
                "content": SUMMARY_PROMPT.format(question=query, context=context),
            }
        ]
        response = self.llm_model.generate_response(messages)
        return response


SUMMARY_PROMPT = """
You are an assistant for summarization tasks. Use the following pieces of retrieved context to summarize the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

# Context: 
{context}

# Question: 
{question}
"""


PROMPT = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
