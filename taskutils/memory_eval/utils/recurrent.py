from .aio import get_async_client
from utils import extract_solution
from .envs import URL, API_KEY, RECURRENT_CHUNK_SIZE, RECURRENT_MAX_NEW, RECURRENT_MAX_CONTEXT_LEN

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and format your response as follows "Therefore, the answer is (insert answer here)".

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    # assert max_length > 50, "max_length must be greater than 50"
    if not len(string) > max_length:
        return string
    target_len = max_length - len('\n\n...(truncated)\n\n')
    return string[:target_len//2] + '\n\n...(truncated)\n\n' + string[-target_len//2:]


async def async_query_llm(item, model, tokenizer, temperature=0.7, top_p=0.95, stop=None):
    idx = item["_id"]
    context = item["context"].strip()
    prompt = item['input'].strip()
    session = await get_async_client()
    async with session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        if len(input_ids) > max_len:
            import loguru; loguru.logger.warning(f"[Truncated] {len(input_ids)=} > {max_len=}")
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
        chunks = []
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk_ids = input_ids[i:i+RECURRENT_CHUNK_SIZE]
            chunks.append(tokenizer.decode(chunk_ids))
        trace = {
            "schema_version": "recurrent_trace_v1",
            "chunk_count": len(chunks),
            "chunk_size": RECURRENT_CHUNK_SIZE,
            "recurrent_max_new": RECURRENT_MAX_NEW,
            "steps": [],
        }
        memory = NO_MEMORY
        for i, chunk_text in enumerate(chunks):
            msg = TEMPLATE.format(prompt=prompt, chunk=chunk_text, memory=memory)
            step_trace = {
                "step_index": i,
                "prompt": clip_long_string(msg),
                "memory_before": clip_long_string(memory),
            }
            # if idx == 0:
            #     print("user:")
            #     print(clip_long_string(msg))
            try:
                async with session.post(
                    url=URL + "/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json=dict(model=model,
                        messages=[{"role": "user", "content": msg}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=RECURRENT_MAX_NEW,
                        chat_template_kwargs={"enable_thinking": False},
                    )
                ) as resp:
                    status = resp.status
                    if status!= 200:
                        print(f"{status=}, {model=}")
                        return ''
                    data = await resp.json()
                    memory, _ = extract_solution(data['choices'][0]['message']['content'])
                    step_trace["output"] = clip_long_string(data['choices'][0]['message']['content'])
                    step_trace["memory_after"] = clip_long_string(memory)
                    trace["steps"].append(step_trace)
                    # if idx == 0:
                    #     print("assistant:")
                    #     print(clip_long_string(memory))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback
                traceback.print_exc()
                return ''
        msg = TEMPLATE_FINAL.format(prompt=prompt, memory=memory)
        # if idx == 0:
        #     print("user:")
        #     print(clip_long_string(msg))
        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(model=model,
                    messages=[{"role": "user", "content": msg}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW,
                    chat_template_kwargs={"enable_thinking": False},
                )
            ) as resp:
                status = resp.status
                if status!= 200:
                    print(f"{status=}, {model=}")
                    return ''
                data = await resp.json()
                final_response = data['choices'][0]['message']['content']
                trace["final"] = {
                    "prompt": clip_long_string(msg),
                    "output": clip_long_string(final_response),
                    "memory": clip_long_string(memory),
                }
                # if idx == 0:
                #     print("assistant:")
                #     print(clip_long_string(data['choices'][0]['message']['content']))
                return {
                    "response": final_response,
                    "trace": trace,
                }
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
        return ''
