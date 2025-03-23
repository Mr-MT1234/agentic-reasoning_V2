from tavily import TavilyClient
from nano_graphrag import GraphRAG
from .agent import Agent
from sou.models.generation_model import Model
from sou.prompt.prompt_manager import get_planning_prompt
from sou.prompt.utils.extract_pattern import extract_search_plan

class SearchAgent(Agent):
    """
    A research agent that can retrieve information from a language model
    and store relevant details into a knowledge graph.
    """

    def __init__(
        self,
        tavily_client: TavilyClient,
        name: str = "SearchAgent",
        knowledge_graph: GraphRAG = None,
        top_k: int = 10,
        threshold: float = 0.5,
        
    ) -> None:
        self.name = name
        self.tavily_client = tavily_client
        self.knowledge_graph = knowledge_graph
        self.top_k = top_k
        self.threshold = threshold

    def gather_information(self, query: str) -> str:
        """
        Obtain information from a Large Language Model (LLM).
        """
        print(f"[{self.name}] Gathering information for query: {query}")
        response = self.tavily_client.search(query, include_raw_content=True)
        response = response["results"][: self.top_k]
        for result in response:
            if result["score"] < self.threshold:
                response.remove(result)

        return response

    def analyze_and_store(self, data: str) -> None:
        """
        Perform analysis of the retrieved data and store relevant findings
        into the knowledge graph. Implementation of how your data is parsed,
        summarized, or broken down into graph relations is entirely up to you.
        """
        print(f"[{self.name}] Analyzing data:\n{data}")

        # Analyze the data and store relevant details into the knowledge graph
        # TODO: Implement the analysis logic here

        if self.knowledge_graph is not None:
            self.knowledge_graph.insert_data(data)
            print(f"[{self.name}] Stored data in KnowledgeGraph")

    def run(self, query: str, deep_search:bool=False):
        """
        The main loop for the agent if you want a single method that performs
        both gather + analyze/store steps.
        """
        if deep_search:
            # TODO
            pass
        info = self.gather_information(query)
        self.analyze_and_store([i["content"] for i in info])
        return info


    def deep_search(self, query: str, model: Model, max_search_limit:int=3, broaden:bool=False)-> str:
        """
        deep search with a planning model
        """
        prompt = get_planning_prompt(query,max_search_limit=max_search_limit,broaden=broaden)
        planning = model.generate_response_from_prompt(prompt=prompt)
        search_plan = extract_search_plan(planning,max_search_limit=max_search_limit,broaden=broaden)
        #TODO: complete deep search

        return search_plan