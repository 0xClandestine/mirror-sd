"""Chat template utilities for Qwen3/Qwen3.5 models with DFlash.

Qwen3 defaults to "thinking mode" (outputs think tags before answering).
The DFlash draft model was trained with thinking DISABLED, so the target
must also run without thinking for good acceptance.

Usage:
    prompt = format_prompt(tokenizer, "What is 15% of 200?")
    tokens = tokenizer.encode(prompt)
"""

NO_THINK_SYSTEM = "/no_think"


def format_prompt(tokenizer, user_message: str, system: str = NO_THINK_SYSTEM) -> str:
    """Apply chat template with thinking disabled for DFlash compatibility.

    The /no_think system prompt tells Qwen3/Qwen3.5 to skip thinking.
    For Qwen3.5+, we also strip trailing think-end tags that
    enable_thinking=False injects, since the DFlash draft model expects
    the prompt to end at the assistant prefix.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    try:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    # Strip trailing think tags that enable_thinking=False adds
    # DFlash draft expects prompt to end at <|im_start|>assistant\n
    import re
    result = re.sub(r'(<\|im_start\|>assistant\n).*', r'\1', result)
    return result


def get_stop_token_ids(tokenizer) -> list:
    """Get stop token IDs including EOS and chat-format end tokens."""
    ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(tokenizer.eos_token_id)
        else:
            ids.add(tokenizer.eos_token_id)
    return list(ids)
