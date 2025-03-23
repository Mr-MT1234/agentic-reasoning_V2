from dataclasses import dataclass
from typing import Optional

@dataclass
class ReasoningSettings:
    model_name: str = "groq/llama3-70b-8192"
    code_model_name: str = "groq/llama3-70b-8192"
    use_rag_agent: bool = True
    use_code_agent: bool = True
    use_search_agent: bool = True
    forcing_search: bool = False
    working_dir: Optional[str] = None
    forcing_search: bool = False
    deep_search: bool = False

