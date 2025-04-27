# extract the research query and code query
import os
from dataclasses import dataclass
from enum import Enum

from sou.agents.agent import Agent
from sou.agents.coding_agent import CodingAgent
from sou.models.embedding_model import EmbeddingModel
from sou.models.generation_model import Model


@dataclass
class ReasoningSettings:
    reasoning_model: str
    coding_model: str


CODING_BEGIN = "<coding_assistant_begin>"
CODING_END = "<coding_assistant_end>"
KNOWLEDGE_BEGIN = "<knowledge_assistant_begin>"
KNOWLEDGE_END = "<knowledge_assistant_end>"


class Entent(Enum):
    CallCodingAgent = 0
    CallKnowledgeAgent = 1


def run_reasoning(prompt, coding_agent: CodingAgent, knowledge_agent: None):
    prompt = ""

    generation_model = Model(model_name="reasoning_model")
    output = generation_model.generate_response_from_prompt(
        prompt,
        stop_tokens=[CODING_BEGIN, KNOWLEDGE_END],
    )

    if CODING_BEGIN in output:
        entent = Entent.CallCodingAgent
        start_query = output.find(CODING_BEGIN) + len(CODING_BEGIN)
        end_query = output.find(CODING_END, start_query)

        if end_query == -1:
            output += CODING_BEGIN
            end_query = len(output)

        query = output[start_query:end_query]
    elif KNOWLEDGE_BEGIN in output:
        entent = Entent.CallKnowledgeAgent
        start_query = output.find(KNOWLEDGE_BEGIN) + len(KNOWLEDGE_BEGIN)
        end_query = output.find(KNOWLEDGE_END, start_query)

        if end_query == -1:
            output += KNOWLEDGE_END
            end_query = len(output)

        query = output[start_query:end_query]

    match entent:
        case Entent.CallCodingAgent:
            return coding_agent.generate_code()
        case Entent.CallKnowledgeAgent:
            pass
        case None:
            pass

    return sequence


def run_reasoning_loop(
    prompt: str,
    agent_list: dict[str, Agent] = None,
    reasoning_settings: ReasoningSettings = None,
):
    """
    Loop to complete reasoning over multiple steps.
    """
    sequence = Sequence(prompt)

    if agent_list is None:
        agent_list = initialize_agents(reasoning_settings)

    print(f"Forcing search: {reasoning_settings.forcing_search}")
    if reasoning_settings.forcing_search:
        search_agent = agent_list.get("search_agent")
        _ = search_agent.run(
            sequence.question, deep_search=reasoning_settings.deep_search
        )

    if reasoning_settings.use_rag_agent:
        rag_agent = agent_list.get("rag_agent")
        if reasoning_settings.forcing_search:
            context = rag_agent.summary_context(sequence.question)
            sequence.add_context(context)

    while not sequence.finished:
        sequence = run_reasoning(agent_list, reasoning_settings, sequence)

    if reasoning_settings.post_processing:
        llm_model = Model(model_name=reasoning_settings.model_name)
        sequence.final_result = post_processing(sequence, llm_model)

    return sequence


def initialize_agents(reasoning_settings: ReasoningSettings) -> dict[str, Agent]:
    """
    Initialize agents based on reasoning settings.
    """
    agent_list = {}

    rag_agent = None

    if reasoning_settings.use_rag_agent:
        embedding_model = EmbeddingModel(
            model_name=reasoning_settings.rag_embedding_model_name
        )
        llm_model = Model(model_name=reasoning_settings.rag_generation_model_name)
        rag_agent = RAGAgent(llm_model=llm_model, embedding_model=embedding_model)

    if reasoning_settings.use_search_agent or reasoning_settings.forcing_search:
        tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        search_agent = SearchAgent(
            tavily_client=tavily_client, knowledge_graph=rag_agent
        )
        agent_list["search_agent"] = search_agent

    if reasoning_settings.use_code_agent:
        code_model = Model(reasoning_settings.code_model_name)
        code_agent = CodeAgent(model=code_model)
        agent_list["code_agent"] = code_agent

    agent_list["rag_agent"] = rag_agent  # always set, even if None

    return agent_list


def post_processing(sequence: Sequence, llm_model: Model):
    """
    Post-processing of the reasoning output.
    """
    PROMPT = """
    Given the following context:
    
    '''
    {context}
    '''


    answer the question: {query}.
    """
    if sequence.final_result is not None:
        return sequence.final_result
    else:
        messages = PROMPT.format(context=sequence.output, query=sequence.question)
        response = llm_model.generate_response_from_prompt(messages)
        return response
