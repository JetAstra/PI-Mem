from pathlib import Path

import loguru

__registered_tokenizer__ = {
    "Qwen2TokenizerFast": "Qwen/Qwen2.5-0.5B-Instruct",
}
__template_dir = Path(__file__).parent


def _should_use_qwen3_5_template(tokenizer) -> bool:
    name_or_path = str(getattr(tokenizer, "name_or_path", "")).lower()
    return "qwen3.5" in name_or_path or "qwen3_5" in name_or_path


def set_chat_template(tokenizer):
    """
    For Qwen2TokenizerFast and Qwen3.5:
    1. add {% generation %} block to support assistant masks
    2. keep rendered text aligned with the model's original chat template
    3. preserve unfinished assistant turns without appending eos
    """
    name = type(tokenizer).__name__
    if _should_use_qwen3_5_template(tokenizer):
        loguru.logger.debug(f"[MemAgent][ChatTemplate] Using Qwen3.5 template for tokenizer `{name}`")
        with open(__template_dir / "Qwen3_5.j2", "r") as f:
            tokenizer.chat_template = f.read()
        return tokenizer

    if name not in __registered_tokenizer__:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
        raise ValueError(
            f"tokenizer {name} not registered and does not provide a chat_template"
        )
    with open(__template_dir / f"{name}.j2", "r") as f:
        tokenizer.chat_template = f.read()
    return tokenizer
