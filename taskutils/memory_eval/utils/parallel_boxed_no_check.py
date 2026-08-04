import asyncio

from .aio import get_async_client
from .envs import (
    API_KEY,
    PARALLEL_MAX_PASSES,
    PARALLEL_MERGE_MAX_TOKENS,
    RECURRENT_CHUNK_SIZE,
    RECURRENT_MAX_CONTEXT_LEN,
    RECURRENT_MAX_NEW,
    URL,
)
from .parallel_boxed import (
    NO_MEMORY,
    TEMPLATE_FINAL_BOXED,
    TEMPLATE_FINAL_BOXED_WITH_ANSWER_PREFIX,
    TEMPLATE_MERGE,
    TEMPLATE_MERGE_WITH_ANSWER_PREFIX,
    _append_trace_record,
    _build_merge_memories,
    _query_single,
    clip_long_string,
)


TEMPLATE_CHUNK_NO_CHECK = """You are presented with a problem, a section of an article, and a global memory summarizing previously gathered information. Please read the section carefully and summarize the information in this section that may be useful for answering the problem. Preserve relevant details from the current section.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Current section summary:
"""


TEMPLATE_CHUNK_NO_CHECK_WITH_ANSWER_PREFIX = """You are presented with a problem, a section of an article, and a global memory summarizing previously gathered information. Please read the section carefully and summarize the information in this section that may be useful for answering the problem. Preserve relevant details from the current section.

<problem>
{prompt}

{answer_prefix}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Current section summary:
"""


async def async_query_llm(
    item,
    model,
    tokenizer,
    temperature=0.7,
    top_p=0.95,
    stop=None,
    trace_path=None,
    use_answer_prefix=False,
):
    idx = item["_id"]
    context = item["context"].strip()
    prompt = item["input"].strip()
    trace_record = {
        "_id": idx,
        "input": clip_long_string(prompt),
        "context_char_len": len(context),
        "passes": [],
        "final": {},
        "chunk_strategy": "summarize_all_no_check",
    }
    session = await get_async_client()
    async with session:
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        trace_record["context_token_len_before_truncation"] = len(input_ids)
        trace_record["context_truncated"] = len(input_ids) > max_len
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        trace_record["context_token_len_after_truncation"] = len(input_ids)

        chunks = []
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk_ids = input_ids[i : i + RECURRENT_CHUNK_SIZE]
            chunks.append(tokenizer.decode(chunk_ids))
        trace_record["num_chunks"] = len(chunks)

        global_memory = NO_MEMORY
        converged = False
        passes_used = 0

        for pass_num in range(PARALLEL_MAX_PASSES):
            passes_used = pass_num + 1
            pass_trace = {
                "pass_num": pass_num,
                "global_memory_before": clip_long_string(global_memory),
                "chunk_phase": {"chunks": []},
                "new_memories": [],
            }

            answer_prefix = item.get("answer_prefix", "")
            if use_answer_prefix and answer_prefix.strip():
                chunk_msgs = [
                    TEMPLATE_CHUNK_NO_CHECK_WITH_ANSWER_PREFIX.format(
                        prompt=prompt,
                        chunk=chunk,
                        memory=global_memory,
                        answer_prefix=answer_prefix,
                    )
                    for chunk in chunks
                ]
            else:
                chunk_msgs = [
                    TEMPLATE_CHUNK_NO_CHECK.format(
                        prompt=prompt,
                        chunk=chunk,
                        memory=global_memory,
                    )
                    for chunk in chunks
                ]

            chunk_tasks = [
                _query_single(
                    session, model, msg, temperature, top_p, RECURRENT_MAX_NEW
                )
                for msg in chunk_msgs
            ]
            chunk_results = await asyncio.gather(*chunk_tasks)

            new_memories = []
            for chunk_idx, (chunk, msg, r) in enumerate(
                zip(chunks, chunk_msgs, chunk_results)
            ):
                has_output = r is not None and bool(str(r).strip())
                pass_trace["chunk_phase"]["chunks"].append(
                    {
                        "chunk_index": chunk_idx,
                        "chunk": clip_long_string(chunk, 1000),
                        "request": clip_long_string(msg),
                        "response": clip_long_string(r) if r else None,
                        "has_output": has_output,
                    }
                )
                if has_output:
                    new_memories.append(r)
            pass_trace["new_memories"] = [clip_long_string(mem) for mem in new_memories]

            if not new_memories:
                converged = True
                pass_trace["merge_phase"] = None
                pass_trace["global_memory_after"] = clip_long_string(global_memory)
                trace_record["passes"].append(pass_trace)
                break

            merge_memories = _build_merge_memories(global_memory, new_memories)
            answer_prefix = item.get("answer_prefix", "")
            if use_answer_prefix and answer_prefix.strip():
                merge_msg = TEMPLATE_MERGE_WITH_ANSWER_PREFIX.format(
                    prompt=prompt,
                    memories=merge_memories,
                    answer_prefix=answer_prefix,
                )
            else:
                merge_msg = TEMPLATE_MERGE.format(
                    prompt=prompt,
                    memories=merge_memories,
                )

            merged = await _query_single(
                session, model, merge_msg, temperature, top_p, PARALLEL_MERGE_MAX_TOKENS
            )
            pass_trace["merge_phase"] = {
                "memories": clip_long_string(merge_memories),
                "request": clip_long_string(merge_msg, max_length=4000),
                "response": clip_long_string(merged) if merged else None,
            }
            if merged is not None:
                global_memory = merged
            pass_trace["global_memory_after"] = clip_long_string(global_memory)
            trace_record["passes"].append(pass_trace)

        answer_prefix = item.get("answer_prefix", "")
        if use_answer_prefix and answer_prefix.strip():
            final_msg = TEMPLATE_FINAL_BOXED_WITH_ANSWER_PREFIX.format(
                prompt=prompt,
                memory=global_memory,
                answer_prefix=answer_prefix,
            )
            trace_record["final"]["answer_prefix"] = answer_prefix
        else:
            final_msg = TEMPLATE_FINAL_BOXED.format(prompt=prompt, memory=global_memory)
        trace_record["final"]["request"] = clip_long_string(final_msg)
        trace_record["final"]["memory"] = clip_long_string(global_memory)

        try:
            async with session.post(
                url=URL + "/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=dict(
                    model=model,
                    messages=[{"role": "user", "content": final_msg}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=RECURRENT_MAX_NEW,
                    chat_template_kwargs={"enable_thinking": False},
                ),
            ) as resp:
                if resp.status != 200:
                    print(f"status={resp.status}, {model=}")
                    result = {
                        "response": "",
                        "parallel_passes_used": passes_used,
                        "parallel_converged": converged,
                        "parallel_max_passes": PARALLEL_MAX_PASSES,
                    }
                    trace_record["final"]["response"] = ""
                    trace_record["final"]["http_status"] = resp.status
                    trace_record.update(result)
                    await _append_trace_record(trace_path, trace_record)
                    result["trace"] = trace_record
                    return result
                data = await resp.json()
                response = data["choices"][0]["message"]["content"]
                result = {
                    "response": response,
                    "parallel_passes_used": passes_used,
                    "parallel_converged": converged,
                    "parallel_max_passes": PARALLEL_MAX_PASSES,
                }
                trace_record["final"]["response"] = clip_long_string(response)
                trace_record["final"]["http_status"] = resp.status
                trace_record.update(result)
                await _append_trace_record(trace_path, trace_record)
                result["trace"] = trace_record
                return result
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            import traceback

            traceback.print_exc()
            trace_record["final"]["exception"] = repr(e)
        result = {
            "response": "",
            "parallel_passes_used": passes_used,
            "parallel_converged": converged,
            "parallel_max_passes": PARALLEL_MAX_PASSES,
        }
        trace_record["final"].setdefault("response", "")
        trace_record.update(result)
        await _append_trace_record(trace_path, trace_record)
        result["trace"] = trace_record
        return result
