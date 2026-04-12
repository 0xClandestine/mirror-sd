"""Chat template utilities for Qwen3/Qwen3.5 models with DFlash.

Qwen3 defaults to "thinking mode" (outputs think tags before answering).
The DFlash draft model was trained with thinking DISABLED, so the target
must also run without thinking for good acceptance.

Usage:
    prompt = format_prompt(tokenizer, "What is 15% of 200?")
    tokens = tokenizer.encode(prompt)
"""

NO_THINK_SYSTEM = "/no_think"


def format_prompt(tokenizer, user_message: str, system: str = NO_THINK_SYSTEM, enable_thinking: bool = False) -> str:
    """Apply chat template.

    By default thinking is disabled for DFlash compatibility.
    Set enable_thinking=True to enable Qwen3/Qwen3.5 thinking mode.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    try:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        result = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    if not enable_thinking:
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
