from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sou.prompt.config import (
    BEGIN_CODE_QUERY,
    BEGIN_CODE_RESULT,
    BEGIN_MIND_MAP_QUERY,
    BEGIN_MIND_MAP_RESULT,
    BEGIN_SEARCH_QUERY,
    BEGIN_SEARCH_RESULT,
    END_CODE_QUERY,
    END_CODE_RESULT,
    END_MIND_MAP_QUERY,
    END_MIND_MAP_RESULT,
    END_SEARCH_QUERY,
    END_SEARCH_RESULT,
)
from sou.prompt.prompt_manager import get_basic_generation_prompt
from sou.prompt.utils.extract_pattern import extract_between


@dataclass
class Sequence:
    question: str
    prompt: str = ""
    output: str = ""
    history: List[str] = field(default_factory=list)
    finished: bool = False
    final_result: Optional[str] = None

    def __post_init__(self):
        if not self.prompt:  # Only set it if not already provided
            self.prompt = get_basic_generation_prompt(self.question)

    def set_output(self, output: str):
        self.output = output

        # Define your tag pairs
        tag_pairs = [
            (BEGIN_SEARCH_QUERY, END_SEARCH_QUERY),
            (BEGIN_CODE_QUERY, END_CODE_QUERY),
            (BEGIN_MIND_MAP_QUERY, END_MIND_MAP_QUERY),
        ]

        search_query = code_query = rag_query = None

        # Process each tag pair
        for begin_tag, end_tag in tag_pairs:
            if begin_tag in output:
                result = extract_between(output, begin_tag, end_tag)

                # If only the begin tag was found, append the end tag to output
                if end_tag not in output[output.find(begin_tag) + len(begin_tag) :]:
                    self.output += end_tag

                # Assign result to appropriate variable
                if begin_tag == BEGIN_SEARCH_QUERY:
                    search_query = result
                elif begin_tag == BEGIN_CODE_QUERY:
                    code_query = result
                elif begin_tag == BEGIN_MIND_MAP_QUERY:
                    rag_query = result

        if search_query is None and code_query is None and rag_query is None:
            self.finished = True

        return search_query, code_query, rag_query

    def complete_output(self, answer: str, answer_type: str):
        """
        Completes the output of the sequence with the given answer.
        @param answer: The answer to the question.
        @param answer_type: The type of the agent that provided the answer. It can be "search", "code", or "rag".
        """

        if answer_type == "search":
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{answer}{END_SEARCH_RESULT}\n\n"
        elif answer_type == "code":
            append_text = f"\n\n{BEGIN_CODE_RESULT}{answer}{END_CODE_RESULT}\n\n"

        elif answer_type == "rag":
            append_text = (
                f"\n\n{BEGIN_MIND_MAP_RESULT}{answer}{END_MIND_MAP_RESULT}\n\n"
            )

        self.output += append_text

    def upd_with_output(self):
        self.history.append(self.output)
        self.prompt = (
            self.prompt + f"\n\nOutput of Step {len(self.history)}: {self.output}"
        )

    def add_context(self, context):
        self.prompt = self.prompt + f"\n\nContext: {context}"
