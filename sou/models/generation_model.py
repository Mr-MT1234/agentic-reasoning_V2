from litellm import completion
from transformers import AutoModel, AutoTokenizer


class Model:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name

    def generate_response(
        self,
        messages: list,
        temperature: int = 0.7,
        top_p: float = None,
        max_tokens: int = None,
        stop_tokens: list = None,
        frequency_penalty: float = None,
    ) -> str:
        # print number of token with autotokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("===========token count", len(tokenizer.tokenize(messages[0]["content"])))
        print(f"Prompt : \n {messages[0]['content']}")

        if self.model_name.startswith("ollama/"):
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_tokens,
                api_base="http://localhost:11434",
                frequency_penalty=frequency_penalty,
            )

        else:
            response = completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop_tokens,
                frequency_penalty=frequency_penalty,
            )

        response = response.choices[0].message.content

        print(f"Response : \n {response}")

        return response

    def generate_response_from_prompt(
        self,
        prompt: str,
        temperature: int = 0.7,
        top_p: float = None,
        max_tokens: int = None,
        stop_tokens: list = None,
        frequency_penalty: float = None,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_response(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
            frequency_penalty=frequency_penalty,
        )
