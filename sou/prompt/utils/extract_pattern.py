import re

import litellm


def extract_between(text: str, start_tag: str, end_tag: str):
    """
    Extract text between two tags.
    If only start_tag is found, return all text after start_tag and append end_tag.
    """
    if start_tag in text:
        start_pos = text.find(start_tag) + len(start_tag)
        if end_tag in text[start_pos:]:
            end_pos = text.find(end_tag, start_pos)
            return text[start_pos:end_pos].strip()
        else:
            # Only found start_tag but not end_tag
            # Return all text after start_tag
            extracted = text[start_pos:].strip()
            return extracted
    return None


def extract_boxed(text):
    try:
        return re.findall(r"\\boxed\{(.*?)\}", text)[-1]
    except Exception:
        print("No chice found")
        return []


def extract_search_plan(text: str, max_search_limit: int, broaden: bool = False):
    # Pattern to match each subtopic and its phrases
    pattern = r"\[Subtopic \d+: (.*?)\]:\s*((?:(?!\[Subtopic).)+)"

    # Find all matches
    matches = re.findall(pattern, text)

    # Create dictionary
    search_plan = {}
    for subtopic, phrases_str in matches:
        phrases = re.findall(r'"(.*?)"', phrases_str)
        search_plan[subtopic.strip()] = phrases

    # select max_search_limit topics
    search_plan = {
        k: v
        for k, v in search_plan.items()
        if k in search_plan.keys()[:max_search_limit]
    }

    if not broaden:
        search_plan = {k: v[:1] for k, v in search_plan.items() if len(v) > 1}

    return search_plan


def get_mcq_answer(question: str) -> str:
    prompt = f"""
        You will be given a multiple-choice question about reliability engineering. 
        Choose the correct answer from the options provided. 
        Respond only with a single character: a, b, c, or d. No explanations.
        
        {question}
        """

    response = litellm.completion(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o",
        temperature=0.0,
    )

    return response["choices"][0]["message"]["content"].strip()
