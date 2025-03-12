from transformers import AutoTokenizer


def initialize_tokenizer(model_name: str, args: dict = None):
    """Initialize the Language Model."""
    if model_name == "gpt-4o":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return tokenizer