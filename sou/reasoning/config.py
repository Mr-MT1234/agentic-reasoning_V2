from dataclasses import dataclass
from typing import Optional

@dataclass
class ReasoningSettings:
    model_name: str = "groq/llama3-70b-8192"
    code_model_name: str = "groq/llama3-70b-8192"
    rag_embedding_model_name: str = "Alibaba-NLP/gte-base-en-v1.5"
    rag_generation_model_name: str = "groq/llama3-70b-8192"
    use_rag_agent: bool = True
    use_code_agent: bool = True
    use_search_agent: bool = True
    forcing_search: bool = False
    working_dir: Optional[str] = None
    deep_search: bool = False
    post_processing: bool = True

