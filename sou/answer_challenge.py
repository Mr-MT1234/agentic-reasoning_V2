# extract the research query and code query
import os
from dataclasses import dataclass
from enum import Enum

from agents.agent import Agent
from agents.coding_agent import CodingAgent
from agents.expert_agent import KnowledgeAgent
from models.embedding_model import EmbeddingModel
from models.generation_model import Model

from prompt.prompts import get_hard_question_instruction
from prompt.special_tokens import *


@dataclass
class ReasoningSettings:
    reasoning_model: str
    coding_model: str


class Entent(Enum):
    CallCodingAgent = 0
    CallKnowledgeAgent = 1
    GiveUp = 2
main_model = 'gpt-4o-mini'
expert_model = 'gpt-4o-mini'
def run_reasoning(prompt) -> tuple[Entent, str]:
    generation_model = Model(model_name=main_model)
    output = generation_model.generate_response_from_prompt(
        prompt,
        stop_tokens=[CODING_QUERY_END, KNOWLEDGE_QUERY_END],
    )

    print("output",output)

    if CODING_QUERY_BEGIN in output:
        start_query = output.find(CODING_QUERY_BEGIN) + len(CODING_QUERY_BEGIN)
        end_query = output.find(CODING_QUERY_END, start_query)

        if end_query == -1:
            output += CODING_QUERY_END
            end_query = len(output)

        entent = Entent.CallCodingAgent
        instructions = output[start_query:end_query]
        return entent, instructions
    elif KNOWLEDGE_QUERY_BEGIN in output:
        entent = Entent.CallKnowledgeAgent
        start_query = output.find(KNOWLEDGE_QUERY_BEGIN) + len(KNOWLEDGE_QUERY_BEGIN)
        end_query = output.find(KNOWLEDGE_QUERY_END, start_query)

        if end_query == -1:
            output += KNOWLEDGE_QUERY_END
            end_query = len(output)

        query = output[start_query:end_query]
        return entent, query
    
    else:
        return Entent.GiveUp, ""

def run_reasoning_loop(
    challenge: str,
):
    ollama_model = Model(model_name=main_model)
    coding_agent = CodingAgent(ollama_model)
    knowledge_agent = KnowledgeAgent(Model(model_name=expert_model))
    final_output = None
    prompt = get_hard_question_instruction(challenge)
    while True:
        print("reasoning")
        entent, query = run_reasoning(prompt)

        if entent == Entent.CallKnowledgeAgent:
            print(f"querying the knowledge agent for: {query}")
            answer = knowledge_agent.generate_answer(query)
            prompt += f"\n {KNOWLEDGE_RESULT_BEGIN} {answer} {KNOWLEDGE_RESULT_END}"

        if entent == Entent.GiveUp:
            print("The model gave up trying")
            break
        
        if entent == Entent.CallCodingAgent:
            print(f"querying the coding agent for: {query}")
            final_output = coding_agent.generate_code(challenge, query)
            break

    print(f"{final_output = }")


if __name__ == "__main__":
    challenge = """
def condition_failure_speed(failure_rate: callable, t: float) -> float:
    ''' Given a callable failure_rate, which represents the failure rate function of a lifetime distribution, calculate the speed of failure probability for the samples that suvive up to t.
    Example:
    >>> condition_failure_speed(lambda x: x, 10)
    10
    '''
"""

    print(challenge)
    run_reasoning_loop(challenge)

    
        