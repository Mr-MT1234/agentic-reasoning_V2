import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .agent import Agent


class RAGAgent(Agent):
    def __init__(
        self,
        name: str = "rag_agent",
        llm_model = None,
        embedding_model_path: str = "Alibaba-NLP/gte-base-en-v1.5",
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.documents = []
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
        self.model = AutoModel.from_pretrained(
            embedding_model_path, trust_remote_code=True
        ).to(device)
        self.llm_model = llm_model

    def purge_document_base(self):
        self.documents = []

    def store_documents(self, documents:list[str]):
        batch_dict = self.tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = self.model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0].to_list()
        self.documents.extend(embeddings)


    def query(self, query:str):


    def answer_query(self, query: str, store_result: bool = True) -> str:
        print(
            f"[{self.name}] Retrieving context from knowledge graph for answering query: {query}"
        )
        return self.kg.query(query)

    def summary_prev_reasoning(self, query: str) -> str:
        print(f"[{self.name}] Retrieving context from knowledge graph for: {query}")
        return self.kg.query(
            f"Summarize the reasoning process of this query: {query}, be short and clear."
        )

    def summary_context(self, query: str) -> str:
        # print(f"[{self.name}] Retrieving context from knowledge graph for: {query}")
        return self.kg.query(
            f"Summarize the context of this query: {query}, be short and clear, for a human to understand better the context."
        )

    def insert_data(self, data: str) -> None:
        print(f"[{self.name}] Inserting data into knowledge graph")
        self.kg.insert(data)
