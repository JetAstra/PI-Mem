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
    Not used, since we may want to process reward and other things...
    For Qwen2TokenizerFast
    1. add {% generation %} block to support assistant_mask
    2. fix extra "\n" in the end of assistant's response
    3. if finish_reason is length and it is the final turn, then do not add <|im_end|>
    """
    name = type(tokenizer).__name__
    if _should_use_qwen3_5_template(tokenizer):
        loguru.logger.debug(f"[Recurrent][ChatTemplate] Using Qwen3.5 template for tokenizer `{name}`")
        with open(__template_dir / "Qwen3_5.j2", "r") as f:
            tokenizer.chat_template = f.read()
        return tokenizer

    if name not in __registered_tokenizer__:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
        raise ValueError(
            f"tokenizer {name} not registered and does not provide a chat_template"
        )
    with open(__template_dir / f"{name}.j2" , 'r') as f:
        tokenizer.chat_template = f.read()
    return tokenizer
