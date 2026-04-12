"""Chat template utilities for Qwen3 models with DFlash.

Qwen3-8B defaults to "thinking mode" (outputs <think> tokens before
answering). The DFlash draft model was trained with thinking DISABLED,
so the target must also run without thinking for good acceptance.

Usage:
    prompt = format_prompt(tokenizer, "What is 15% of 200?")
    tokens = tokenizer.encode(prompt)
"""

NO_THINK_SYSTEM = "/no_think"


def format_prompt(tokenizer, user_message: str, system: str = NO_THINK_SYSTEM) -> str:
    """Apply Qwen3 chat template with /no_think to disable thinking mode.

    Returns the formatted prompt string ready for tokenization.
    The assistant prefix is included so the model generates directly.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def get_stop_token_ids(tokenizer) -> list:
    """Get stop token IDs including EOS and chat-format end tokens."""
    ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(tokenizer.eos_token_id)
        else:
            ids.add(tokenizer.eos_token_id)
    return list(ids)
