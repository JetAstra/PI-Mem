import time

from openai import OpenAI

from .aio import get_async_client
from .envs import API_KEY, MAX_INPUT_LEN, MAX_OUTPUT_LEN, URL

template_0shot = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

Question: $Q$

Answer: Therefore, the answer is"""

ANSWER_PREFIX = "Therefore, the answer is"


def _encode_silently(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return encoded["input_ids"]


async def async_query_llm(
    item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None
):
    max_input_tokens = MAX_INPUT_LEN
    max_new_tokens = MAX_OUTPUT_LEN
    context = item["context"]
    prompt = template_0shot.replace("$DOC$", context.strip()).replace(
        "$Q$", item["input"].strip()
    )
    session = await get_async_client()
    async with session:
        input_ids = _encode_silently(tokenizer, prompt)
        if len(input_ids) > max_input_tokens:
            input_ids = (
                input_ids[: max_input_tokens // 2] + input_ids[-max_input_tokens // 2 :]
            )
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
        try:
            payload = dict(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            if stop is not None:
                payload["stop"] = stop
            async with session.post(
                url=URL + "/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
            ) as resp:
                if resp.status != 200:
                    print(f"status={resp.status}, {model=}")
                    return ""
                data = await resp.json()
                text = data["choices"][0]["text"].lstrip()
                return f"{ANSWER_PREFIX} {text}".strip()
        except KeyboardInterrupt as e:
            raise e
        except Exception:
            import traceback

            traceback.print_exc()
        return ""


def query_llm(
    prompt,
    model,
    tokenizer,
    temperature=0.7,
    top_p=0.95,
    max_input_tokens=120000,
    max_new_tokens=10000,
    stop=None,
):
    client = OpenAI(base_url=URL, api_key=API_KEY, timeout=1800)
    input_ids = _encode_silently(tokenizer, prompt)
    if len(input_ids) > max_input_tokens:
        input_ids = (
            input_ids[: max_input_tokens // 2] + input_ids[-max_input_tokens // 2 :]
        )
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            kwargs = dict(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            if stop is not None:
                kwargs["stop"] = stop
            completion = client.completions.create(**kwargs)
            text = completion.choices[0].text.lstrip()
            return f"{ANSWER_PREFIX} {text}".strip()
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print('Error Occurs: "%s"        Retry ...' % (str(e)))
            time.sleep(1)
    print("Max tries. Failed.")
    return ""
