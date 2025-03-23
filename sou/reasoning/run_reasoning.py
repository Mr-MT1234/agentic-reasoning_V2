# extract the research query and code query
from sou.agents.code_agent import CodeAgent
from sou.agents.graphrag_agent import GraphRAGAgent
from sou.agents.search_agent import SearchAgent
import os
from sou.agents.agent import Agent
from sou.models.generation_model import Model
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
        sequence.complete_output(answer=search_result, answer_type="search")

    if reasoning_settings.use_code_agent and code_query is not None:
        code_result = code_agent(code_query)
        sequence.complete_output(answer=code_result, answer_type="code")

    if reasoning_settings.use_rag_agent and rag_query is not None:
        rag_result = rag_agent.answer_query(rag_query)
        sequence.complete_output(answer=rag_result, answer_type="rag")

    sequence.upd_with_output()
    return sequence


def run_reasoning_loop(
    agent_list: dict[str, Agent],
    reasoning_settings: ReasoningSettings,
    sequence: Sequence,
):
    """
    Loop to complete reasoning over multiple steps.
    """
    if reasoning_settings.forcing_search:
        search_agent = agent_list.get("search_agent")
        search_query = search_agent.run(
            sequence.prompt, deep_search=reasoning_settings.deep_search
        )
        sequence.complete_output(answer=search_query, answer_type="search")

    if reasoning_settings.use_rag_agent:
        rag_agent = agent_list.get("rag_agent")
        context = rag_agent.summary_context(sequence.prompt)
        sequence.add_context(context)

    while not sequence.finished:
        sequence = run_reasoning(agent_list, reasoning_settings, sequence)

    return sequence


def initialize_agents(reasoning_settings: ReasoningSettings) -> dict[str, Agent]:
    """
    Initialize agents based on reasoning settings.
    """
    agent_list = {}

    rag_agent = None
    if reasoning_settings.use_rag_agent:
        rag_agent = GraphRAGAgent(working_dir=reasoning_settings.working_dir)

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
