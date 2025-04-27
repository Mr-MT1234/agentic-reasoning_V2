def get_planning_instruction(query: str, MAX_SEARCH_LIMIT: int, broaden: bool = False):
    prompt = f"""
        You are a highly skilled online researcher. Your task is to develop a clear, structured, and efficient search plan to gather accurate and relevant information from the internet. Do not provide the final answers—only design a search strategy. 
        Query: "{query}"

        Guidelines & Constraints:

        Limit the total number of subtopics or search items to {MAX_SEARCH_LIMIT}.

        Break down the query into logical subtopics or key questions to guide the search.
        """

    if broaden:
        prompt += """
        For each subtopic, include:

        Primary search terms or exact phrases likely to yield quality results.

        Alternative phrasings or keyword variations to broaden the scope of search.
        Format your output as follows:

        [Subtopic 1]: [Exact phrase or keyword, Alternative phrasings or keyword variations]
        [Subtopic 2]: [Exact phrase or keyword, Alternative phrasings or keyword variations]
        ...

        Your goal is to maximize search relevance and coverage while keeping the plan concise and targeted. 
        """

    else:
        prompt += """
        For each subtopic, include:

        Primary search terms or exact phrases likely to yield quality results.

        Format your output as follows:

        [Subtopic 1]: [Exact phrase or keyword]
        [Subtopic 2]: [Exact phrase or keyword]
        ...

        Your goal is to maximize search relevance and coverage while keeping the plan concise and targeted. 
        """
    return prompt


def get_hard_question_instruction():
    return (
        "You are a reasoning assistant with the ability to access a local mind map "
        "you solve the user's challenge (query) accurately by providing a pseudo code, and then use the code agent to translate the pseudo code into a python code. Your final answer should contain the complete python code. You have special tools:\n\n"
        "- To transform a pseudo code into a python code, you can propose a code task using: <begin_code_query> your code query here </end_code_query>.\n"
        "The system will write the code and provide it to you in the format <begin_code_result> ...python code... </end_code_result>.\n"
        "Make sure your each code query is self-contained and does not require any external information.\n\n"
        "- To access your reasoning memory, you can query the automatically generated mind map using the following format: <begin_mind_map_query> your query here </end_mind_map_query>.\n"
        "The system will then analyze your previous reasoning and answer your query in the following format: <begin_mind_map_result> ...answer results... </end_mind_map_result>\n\n"
        f"You can repeat calling the tools multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}. The code attempts are unlimited.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example1:\n"
        'Challenge: "def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:\n\n"'
        ''' You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item. You will be given two floats t and delta_t. Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
            Examples:
            >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
            0.6819076271964216
        ''' "\n\n"
        "Assistant thinking steps:\n\n"
        "To solve the challenge, I need to better define a reliability function and to know how to link it to failure probabilities\n\n"
        "Assistant:\n"
        "<begin_mind_map_query>What is a reliability function, and how is the probability of failure related to the reliability function?</end_mind_map_query>\n\n"
        "<begin_mind_map_result>\n"
        "The reliability function R(t) gives the probability that an item survives beyond time t: R(t) = P(T > t)\n"
        "Therefore, the probability of failure before time t is: P(T ≤ t) = 1 - R(t)\n"
        "Similarly, the probability of survival after time (t + delta_t) is: P(T > t + delta_t) = R(t + delta_t)\n"
        "Thus, the probability that the item fails either before t or after t + delta_t is: P(T ≤ t) + P(T > t + delta_t) = (1 - R(t)) + R(t + delta_t)\n"
        "Note: We assume the two events (fail before t, fail after t + delta_t) are disjoint.\n"
        "</end_mind_map_result>\n\n"
        "Now, I fully understand the math.\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Next, I will write the pseudo-code for the solution:\n"
        "Function p_f_interval(reliability, t, delta_t):\n"
        "   fail_before_t = 1 - reliability(t)\n"
        "survive_after_t_plus_delta = reliability(t + delta_t)\n"
        "result = fail_before_t + survive_after_t_plus_delta\n"
        "return result\n\n"
        "Assistant:\n"
        "<begin_code_query> Write a Python function p_f_interval(reliability: callable, t: float, delta_t: float) -> float that:\n"
        "Calculates 1 - reliability(t)\n"
        "Calculates reliability(t + delta_t)\n"
        "Returns the sum of these two quantities. Make sure to import numpy as np. </end_code_query>\n\n"
        "<begin_code_result>\n"
        "import numpy as np\n\n"
        "def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:\n"
        "   fail_before_t = 1 - reliability(t)\n"
        "   survive_after_t_plus_delta = reliability(t + delta_t)\n"
        "   return fail_before_t + survive_after_t_plus_delta\n"
        "<end_code_result>\n\n"
        "Remember:\n"
        "- Use <begin_mind_map_query> to request a search from and end with </end_mind_map_query>.\n"
        "- Use <begin_code_query> to request a python code syntax and end with </end_code_query>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <begin_search_result> ...search results... </end_search_result>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "How do you compute the integral of e^(x^2) dx?"\n'
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<begin_search_query>methods to integrate e^(x^2)</end_search_query>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <begin_search_result> ...search results... </end_search_result>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree."\n'
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<begin_search_query>Minimum Steiner Tree problem in trees</end_search_query>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <begin_search_result> ...search results... </end_search_result>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Who got the first Nobel Prize in Physics?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<begin_search_query>first Nobel Prize in Physics winner</end_search_query>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <begin_search_result> ...search results... </end_search_result>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<begin_search_query>Alice David Lara Croft voice</end_search_query>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<begin_search_query>video game developed by Alice David Lara Croft</end_search_query>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_singleqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <begin_search_result> ...search results... </end_search_result>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <begin_url> url1, url2, ... </end_url>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <begin_full_page> ...full page content... </end_full_page>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Who got the first Nobel Prize in Physics?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<begin_search_query>first Nobel Prize in Physics winner</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results mention several URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<begin_url>http://example.com/first_nobel_physics.html</end_url>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<begin_full_page> ...full page content... </end_full_page>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- Use <begin_url> to request full page content and end with </end_url>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_multiqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <begin_search_result> ...search results... </end_search_result>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <begin_url> url1, url2, ... </end_url>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <begin_full_page> ...full page content... </end_full_page>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"\n'
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<begin_search_query>voice actor of Lara Croft</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results provide names of voice actors for Lara Croft. I need to confirm if Alice David is one of them.\n\n"
        "Assistant:\n"
        "<begin_search_query>Alice David Lara Croft voice</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<begin_search_query>video game developed by Alice David Lara Croft</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results mention the company that developed the video game featuring Alice David as Lara Croft.\n\n"
        "Assistant:\n"
        "<begin_url>http://example.com/lara_croft_voice_actor.html, http://example.com/game_developer.html</end_url>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<begin_full_page> ...full page content... </end_full_page>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- Use <begin_url> to request full page content and end with </end_url>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_gpqa_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <begin_search_result> ...search results... </end_search_result>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <begin_url> url1, url2, ... </end_url>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <begin_full_page> ...full page content... </end_full_page>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "What is the energy range of pp III neutrinos?"\n'
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<begin_search_query>pp III neutrino energy spectrum</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<begin_url>http://example.com/ppIII_neutrino.html</end_url>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<begin_full_page> ...full page content... </end_full_page>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- Use <begin_url> to request full page content and end with </end_url>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_math_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s math-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <begin_search_result> ...search results... </end_search_result>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <begin_url> url1, url2, ... </end_url>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <begin_full_page> ...full page content... </end_full_page>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "How do you compute the integral of e^(x^2) dx?"\n'
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<begin_search_query>methods to integrate e^(x^2)</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<begin_url>http://example.com/integration_e_x_squared.html</end_url>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<begin_full_page> ...full page content... </end_full_page>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- Use <begin_url> to request full page content and end with </end_url>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_code_rag_agent_instruction(MAX_SEARCH_LIMIT, MAX_URL_FETCH):
    return (
        "You are a reasoning assistant with the ability to perform web searches and retrieve webpage content to help "
        "you answer the user’s programming-related question accurately. You have special tools:\n\n"
        "- To perform a search: write <begin_search_query> your query here </end_search_query>.\n"
        "Then, the system will call the web search API with your query and return the search results to you in the format <begin_search_result> ...search results... </end_search_result>.\n"
        "  The search results will contain a list of webpages with titles, URLs, and snippets (but not full content).\n\n"
        "- After receiving the search results, if you need more detailed information from one or more specific URLs, write <begin_url> url1, url2, ... </end_url>.\n"
        "  The system will fetch the full page content of those URLs and return it to you as <begin_full_page> ...full page content... </end_full_page>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n"
        f"You can fetch up to {MAX_URL_FETCH} URLs for detailed information.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        'Question: "How do I implement a binary search algorithm in Python?"\n'
        "Assistant thinking steps:\n"
        "- I might need to look up the implementation details of binary search in Python.\n\n"
        "Assistant:\n"
        "<begin_search_query>binary search algorithm implementation in Python</end_search_query>\n\n"
        "(System returns search results)\n\n"
        "Assistant:\n"
        "<begin_search_result> ...search results without full page... </end_search_result>\n\n"
        "Assistant thinks: The search results mention some URLs. I want full details from one of them.\n\n"
        "Assistant:\n"
        "<begin_url>http://example.com/python_binary_search.html</end_url>\n\n"
        "(System returns full page content)\n\n"
        "Assistant:\n"
        "<begin_full_page> ...full page content... </end_full_page>\n\n"
        "Now the assistant has enough info and can continue reasoning.\n\n"
        "Remember:\n"
        "- Use <begin_search_query> to request a web search and end with </end_search_query>.\n"
        "- Use <begin_url> to request full page content and end with </end_url>.\n"
        "- When done retrieving information, continue your reasoning.\n\n"
    )


def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )


def get_basic_task_instruction(question, model_name=None):
    user_prompt = f"""
        Please answer the following question. 
        Question:\n{question}
        """
    return user_prompt


def get_task_instruction_openqa(question, model_name=None):
    if model_name == "qwq":
        user_prompt = (
            "Please answer the following question. "
            "You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
            f"Question:\n{question}\n\n"
        )
    else:
        user_prompt = (
            "Please answer the following question. You should think step by step to solve it.\n\n"
            "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
            f"Question:\n{question}\n\n"
        )
    return user_prompt


def get_task_instruction_math(question, model_name=None):
    if model_name == "qwq":
        user_prompt = (
            "Please answer the following math question. "
            "You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
            f"Question:\n{question}\n\n"
        )
    else:
        user_prompt = (
            "Please answer the following math question. You should think step by step to solve it.\n\n"
            "Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"
            f"Question:\n{question}\n\n"
        )
    return user_prompt


def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == "qwq":
        user_prompt = (
            "Please answer the following multiple-choice question. "
            "You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n"
            f"Question:\n{question}\n\n"
        )
    elif model_name == "llama":
        user_prompt = (
            "Please answer the following multiple-choice question. You should think step by step to solve it.\n\n"
            "Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n"
            f"Question:\n{question}\n\n"
        )
    else:
        user_prompt = (
            "Please answer the following multiple-choice question. You should think step by step to solve it.\n\n"
            "Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n"
            f"Question:\n{question}\n\n"
        )
    return user_prompt


def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == "qwq":
        user_prompt = (
            "Generate a correct Python program that passes all tests for the given problem. "
            "You should provide your final code within a Python code block using triple backticks (```python\n"
            "YOUR_CODE\n"
            "```).\n\n"
            f"Problem Title: {question_title}\n\n"
            f"Problem Statement:\n{question}\n\n"
        )
    else:
        user_prompt = (
            "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. "
            f"You should think step by step to solve it.\n\nQuestion:\n{question}\n\n"
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt
