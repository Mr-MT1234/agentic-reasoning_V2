# extract the research query and code query
from sou.agents.code_agent import CodeAgent
from sou.agents.rag_agent import RAGAgent
from sou.agents.search_agent import SearchAgent
import os
from sou.agents.agent import Agent
from sou.models.generation_model import Model
from sou.models.embedding_model import EmbeddingModel
from .sequence import Sequence
from .config import ReasoningSettings

from tavily import TavilyClient


def run_reasoning(
    agent_list: dict[str, Agent],
    reasoning_settings: ReasoningSettings,
    sequence: Sequence,
):
    """
    Main reasoning function.
    """
    search_agent = agent_list.get("search_agent")
    code_agent = agent_list.get("code_agent")
    rag_agent = agent_list.get("rag_agent")

    generation_model = Model(model_name=reasoning_settings.model_name)
    output = generation_model.generate_response_from_prompt(sequence.prompt)
    search_query, code_query, rag_query = sequence.set_output(output)
    if reasoning_settings.use_search_agent and search_query is not None:
        search_result = search_agent.run(search_query)
        # Generate a new context
        try:
            context = rag_agent.summary_context(search_query)
            sequence.add_context(context)
        except Exception as e:
            sequence.complete_output(answer=search_result, answer_type="search")


    if reasoning_settings.use_code_agent and code_query is not None:
        code_result = code_agent(code_query)
        sequence.complete_output(answer=code_result, answer_type="code")

    if reasoning_settings.use_rag_agent and rag_query is not None:
        rag_result = rag_agent.query(rag_query)
        sequence.complete_output(answer=rag_result, answer_type="rag")

    sequence.upd_with_output()
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

    if reasoning_settings.forcing_search:
        search_agent = agent_list.get("search_agent")
        _ = search_agent.run(
            sequence.question, deep_search=reasoning_settings.deep_search
        )

    if reasoning_settings.use_rag_agent:
        rag_agent = agent_list.get("rag_agent")
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
    Given the following context: {context}, answer the question: {query}.
    """
    if sequence.final_result is not None:
        return sequence.final_result
    else:
        messages = PROMPT.format(context=sequence.output, query=sequence.question)
        response = llm_model.generate_response_from_prompt(messages)
        return response
