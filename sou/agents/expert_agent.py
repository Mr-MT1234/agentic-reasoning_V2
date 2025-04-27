from .agent import Agent
from models.generation_model import Model 

class KnowledgeAgent(Agent):
    def __init__(self, llm_model: Model, name: str="knowledge agent"):
        super().__init__(name)
        self.model = llm_model

    def generate_answer(self, question: str):
        prompt = KNOWLEDGE_PROMPT.format(question)
        output = self.model.generate_response_from_prompt(prompt)
        return output
    

    





KNOWLEDGE_PROMPT = """
You are an expert in the domain of risk and reliability. 
Please answer the following question briefly:
{}
"""