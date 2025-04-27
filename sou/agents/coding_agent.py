from .agent import Agent
from ..models.generation_model import Model 

class CodingAgent(Agent):
    def __init__(self, llm_model: Model, name: str="coding agent"):
        super().__init__(name)
        self.model = llm_model

    def generate_code(self, signature: str, instructions: str):
        prompt = GENERATION_PROMPT.format(signature = signature, instructions = instructions)
        output = self.model.generate_response_from_prompt(prompt)
        return self._extract_code(output)
    
    def _extract_code(self, output:str):
        end = output.rfind('```')
        start = output.rfind('```', 0, end) + 3
        if output[start:].startswith('python'):
            start += len('python')

        return output[start: end]

GENERATION_PROMPT="""
You are a coding assitant tasked with helping scientits write Python code quickly and accuratly.
You will receive a function signature as well as step by step instruction of what the function should do
and your goal is to write the complete function.
- Make sure that the written code does not use any external libraries. The function must be self contained.
- The function signature will be given between the tags <signature_begin> and <signature_end>.
- The instructions to follow will be given between <instructions_begin> and <instructions_end>.
- The final function must be written between '```'
- Be breif.

# Example:
Input: 
Write a function with the following signature
<signature_begin>
def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:
    '''
    You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item.
    You will be given two floats t and delta_t.
    Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
    Examples:
    >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
    0.6819076271964216
    '''
<signature_end>

The function should follow the following instructions:
<instructions_begin>
1. **Given Input**: `reliability` (callable), `t` (float), and `delta_t` (float)
2. **Evaluate CDF at T and T+ΔT**: Calculate `prob_before = reliability(t)` and `prob_after = reliability(t +
delta_t)`
3. **Calculate P**: P is the difference between prob_after and prob_before: `P = prob_after - prob_before`
4. **Return Result**: Return the calculated probability `P` as a float value
<instructions_end>

Output:
```
def p_f_interval(reliability: callable, t: float, delta_t: float) -> float:
    '''
    You will be given a callable reliability, which is a reliability function a lifetime distribution T, representing the lifetime of an item.
    You will be given two floats t and delta_t.
    Your task is to develop a python script to calculate the probability that the item fails either before t or after t + delta_t.
    Examples:
    >>> p_f_interval(lambda x: np.exp(-x), 1, 2)
    0.6819076271964216
    '''
    prob_before = reliability(t)
    prob_after = reliability(t + delta_t)

    P = prob_after - prob_before

    return P
```

# Your Input:
Write a function of the following signature
<signature_begin>
{signature}
<signature_end>

The function should follow the following instructions:
<instructions_begin>
{instructions}
<instructions_end>
"""