import asyncio
import json
import os

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

# Same templates as recurrent/impls/memory.py
TEMPLATE_CHUNK = """You are presented with a problem, a section of an article, and a global memory summarizing previously gathered information. Please read the section carefully and determine whether the section contains new information relevant to answering the problem beyond what is already in the global memory. First, output your judgment in the format <check>yes</check> if there is new information, or <check>no</check> if there is none. Then, if there is new information, extract and list the key details.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Your response:
"""

TEMPLATE_MERGE = """You are presented with a problem and key information extracted from multiple sections of an article. Please consolidate all the information into a single comprehensive memory. Remove redundancies and organize the information clearly, retaining all details relevant to answering the problem.

<problem>
{prompt}
</problem>

<extracted_information>
{memories}
</extracted_information>

Consolidated memory:
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
NO_NEW_INFO_MARKER = "<check>no</check>"
PREVIOUS_MEMORY_LABEL = "[Previous Memory]:"
SECTION_LABEL_TEMPLATE = "[Section {idx}]:"
TRACE_WRITE_LOCK = None


def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    if not len(string) > max_length:
        return string
    target_len = max_length - len("\n\n...(truncated)\n\n")
    return (
        string[: target_len // 2]
        + "\n\n...(truncated)\n\n"
        + string[-target_len // 2 :]
    )


async def _append_trace_record(trace_path: str | None, record: dict):
    """Append one per-item debug trace as a JSONL line."""
    if not trace_path:
        return

    global TRACE_WRITE_LOCK
    if TRACE_WRITE_LOCK is None:
        TRACE_WRITE_LOCK = asyncio.Lock()

    trace_dir = os.path.dirname(trace_path)
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)

    async with TRACE_WRITE_LOCK:
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _query_single(session, model, msg, temperature, top_p, max_tokens):
    """Send a single chat completion request and return raw assistant content."""
    try:
        async with session.post(
            url=URL + "/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=dict(
                model=model,
                messages=[{"role": "user", "content": msg}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": False},
            ),
        ) as resp:
            if resp.status != 200:
                print(f"status={resp.status}, {model=}")
                return None
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip() if isinstance(content, str) else content
    except KeyboardInterrupt as e:
        raise e
    except Exception:
        import traceback

        traceback.print_exc()
        return None


def _is_no_new_info(response: str) -> bool:
    """Check if the model's response contains <check>no</check>."""
    return NO_NEW_INFO_MARKER in response.lower()


def _build_merge_memories(previous_memory: str, new_memories: list[str]) -> str:
    """
    Build the <extracted_information> payload for merge:
    - include previous global memory if it exists
    - append all newly extracted chunk memories with section labels
    """
    parts = []

    if (
        previous_memory
        and previous_memory.strip()
        and previous_memory.strip() != NO_MEMORY
    ):
        parts.append(f"{PREVIOUS_MEMORY_LABEL}\n{previous_memory.strip()}")

    for i, mem in enumerate(new_memories, start=1):
        if mem is None:
            continue
        mem = mem.strip()
        if not mem:
            continue
        parts.append(f"{SECTION_LABEL_TEMPLATE.format(idx=i)}\n{mem}")

    return "\n\n".join(parts) if parts else NO_MEMORY


async def async_query_llm(
    item,
    model,
    tokenizer,
    temperature=0.7,
    top_p=0.95,
    stop=None,
    trace_path=None,
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
    }
    session = await get_async_client()
    async with session:
        # Tokenize and truncate context
        max_len = RECURRENT_MAX_CONTEXT_LEN
        input_ids = tokenizer.encode(context)
        trace_record["context_token_len_before_truncation"] = len(input_ids)
        trace_record["context_truncated"] = len(input_ids) > max_len
        if len(input_ids) > max_len:
            input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        trace_record["context_token_len_after_truncation"] = len(input_ids)

        # Split into chunks
        chunks = []
        for i in range(0, len(input_ids), RECURRENT_CHUNK_SIZE):
            chunk_ids = input_ids[i : i + RECURRENT_CHUNK_SIZE]
            chunks.append(tokenizer.decode(chunk_ids))
        trace_record["num_chunks"] = len(chunks)

        if idx == 0:
            print(f"[parallel] {len(chunks)} chunks, max_passes={PARALLEL_MAX_PASSES}")

        # ---- Multi-pass: chunk (parallel) → merge → repeat ----
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

            # Phase 1: Extract info from ALL chunks in parallel, conditioned on global_memory
            chunk_msgs = [
                TEMPLATE_CHUNK.format(prompt=prompt, chunk=chunk, memory=global_memory)
                for chunk in chunks
            ]
            # if idx == 0:
            #     print(f"--- pass {pass_num} chunk phase ---")
            #     print("user (chunk 0):")
            #     print(clip_long_string(chunk_msgs[0]))

            chunk_tasks = [
                _query_single(
                    session, model, msg, temperature, top_p, RECURRENT_MAX_NEW
                )
                for msg in chunk_msgs
            ]
            chunk_results = await asyncio.gather(*chunk_tasks)

            # Filter: keep only chunks that contain new info
            new_memories = []
            for chunk_idx, (chunk, msg, r) in enumerate(
                zip(chunks, chunk_msgs, chunk_results)
            ):
                has_new_info = r is not None and not _is_no_new_info(r)
                pass_trace["chunk_phase"]["chunks"].append(
                    {
                        "chunk_index": chunk_idx,
                        "chunk": clip_long_string(chunk),
                        "request": clip_long_string(msg),
                        "response": clip_long_string(r) if r else None,
                        "has_new_info": has_new_info,
                    }
                )
                if has_new_info:
                    new_memories.append(r)
            pass_trace["new_memories"] = [clip_long_string(mem) for mem in new_memories]

            if idx == 0:
                print(
                    f"[parallel] pass {pass_num}: {len(new_memories)}/{len(chunks)} chunks had new info"
                )
                for i, mem in enumerate(new_memories):
                    print(f"assistant (new info {i}):")
                    print(clip_long_string(mem))

            # Convergence: no chunk returned new info → stop early
            if not new_memories:
                if idx == 0:
                    print(f"[parallel] Converged at pass {pass_num}, skipping merge")
                converged = True
                pass_trace["merge_phase"] = None
                pass_trace["global_memory_after"] = clip_long_string(global_memory)
                trace_record["passes"].append(pass_trace)
                break

            # Phase 2: Merge previous global_memory + all new chunk memories
            merge_memories = _build_merge_memories(global_memory, new_memories)
            merge_msg = TEMPLATE_MERGE.format(
                prompt=prompt,
                memories=merge_memories,
            )

            if idx == 0:
                print(f"--- pass {pass_num} merge phase ---")
                print("user (merge):")
                print(clip_long_string(merge_msg))

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

            if idx == 0:
                print("assistant (merge):")
                print(clip_long_string(global_memory))

            # if pass_num == PARALLEL_MAX_PASSES - 1:
            #     print(
            #         f"[parallel] Reached max passes ({PARALLEL_MAX_PASSES}), stopping"
            #     )
            #     json_line = {"_id": idx, "input": prompt, "context": context}
            #     import json

            #     with open("fault.jsonl", "a") as f:
            #         f.write(json.dumps(json_line) + "\n")

            trace_record["passes"].append(pass_trace)

        if idx == 0:
            label = (
                "converged"
                if converged
                else f"max_passes={PARALLEL_MAX_PASSES} reached"
            )
            print(f"[parallel] Entering final phase ({label})")

        # ---- Phase 3: Final answer ----
        final_msg = TEMPLATE_FINAL.format(prompt=prompt, memory=global_memory)
        trace_record["final"]["request"] = clip_long_string(final_msg)
        trace_record["final"]["memory"] = clip_long_string(global_memory)
        if idx == 0:
            print("user (final):")
            print(clip_long_string(final_msg))

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
                if idx == 0:
                    print("assistant (final):")
                    print(response)
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
