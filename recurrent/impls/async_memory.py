import asyncio
import logging
import time
from typing import List, Optional, Tuple, Union, Dict
from uuid import uuid4

import math
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
from recurrent.async_utils import ChatCompletionProxy


from recurrent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from recurrent.utils import msg
from verl.protocol import DataProtoItem
from recurrent.generation_manager import _timer
import verl.utils.torch_functional as verl_F


logger = logging.getLogger(__file__)
logger.setLevel("INFO")


from recurrent.impls.memory import (
    MemoryConfig,
    TEMPLATE_CHUNK,
    TEMPLATE_MERGE,
    TEMPLATE_FINAL_BOXED,
    NO_MEMORY,
    build_merge_memories,
)


class AsyncMemoryDataset(RDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        recurrent_config: MemoryConfig,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if data_config.truncation != "middle":
            raise ValueError("AsyncMemoryDataset only support center truncation")
        chunk_size = recurrent_config.chunk_size
        old_max_chunks = recurrent_config.max_chunks
        old_max_prompt_length = data_config.max_prompt_length

        # If both are set, trust prompt-length budget and align max_chunks by ceil.
        if old_max_chunks is not None and old_max_prompt_length is not None:
            new_max_chunks = math.ceil(old_max_prompt_length / chunk_size)
            recurrent_config.max_chunks = new_max_chunks
            data_config.max_prompt_length = new_max_chunks * chunk_size
            logger.info(
                "[AsyncMemoryDataset] recompute by max_prompt_length/chunk_size: "
                f"max_chunks {old_max_chunks} -> {new_max_chunks}, "
                f"max_prompt_length {old_max_prompt_length} -> {data_config.max_prompt_length} "
                f"(chunk_size={recurrent_config.chunk_size})"
            )
        elif old_max_chunks is None and old_max_prompt_length is not None:
            recurrent_config.max_chunks = math.ceil(old_max_prompt_length / chunk_size)
            data_config.max_prompt_length = recurrent_config.max_chunks * chunk_size
            logger.info(
                "[AsyncMemoryDataset] infer max_chunks from max_prompt_length: "
                f"max_chunks None -> {recurrent_config.max_chunks}, "
                f"max_prompt_length {old_max_prompt_length} -> {data_config.max_prompt_length} "
                f"(chunk_size={recurrent_config.chunk_size})"
            )
        elif old_max_chunks is not None:
            data_config.max_prompt_length = old_max_chunks * chunk_size
            logger.info(
                "[AsyncMemoryDataset] infer max_prompt_length from max_chunks: "
                f"max_chunks {recurrent_config.max_chunks}, "
                f"max_prompt_length {data_config.max_prompt_length} "
                f"(chunk_size={recurrent_config.chunk_size})"
            )
        else:
            raise ValueError("Either recurrent_config.max_chunks or data_config.max_prompt_length must be set.")

        self.context_key = recurrent_config.context_key
        logger.info(f"[Check Recurrent Cfg] {recurrent_config}\n")
        super().__init__(
            recurrent_config=recurrent_config,
            data_files=data_files,
            tokenizer=tokenizer,
            data_config=data_config,
            processor=processor,
        )

    @override
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]

        chat = row_dict.pop(self.prompt_key)
        context = row_dict.pop(self.context_key).strip()
        prompt = chat[0]["content"].strip()

        model_inputs = self.tokenizer(
            context, return_tensors="pt", add_special_tokens=False
        )

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,  # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt"] = prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index
        row_dict["sample_uuid"] = str(uuid4())

        return row_dict

    @override
    def get_bactch_keys(self) -> Tuple[List[str], List[str]]:
        # tensor can use 2-deminsional index for chunking.
        # while prompt will not be indexed, so keep it as list.
        return ["context_ids", "context_length"], ["prompt"]


class AsyncMemoryAgent(AsyncRAgent):
    def __init__(
        self,
        proxy: ChatCompletionProxy,
        tokenizer: PreTrainedTokenizer,
        config: RConfig,
        rollout_config: DictConfig,
    ):
        super().__init__(proxy, tokenizer, config, rollout_config)

    @staticmethod
    def _is_trace_sample(sample_index: int) -> bool:
        # Keep logs sparse: only emit stage-progress lines for one sample.
        return sample_index == 0

    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        Multi-pass iterative rollout for a single sample.
        Pass 1: independently extract info from each chunk
        Merge: consolidate all chunk memories into global memory
        Pass 2+: re-extract with global memory context
        Final: generate answer from global memory
        """
        timing_raw = {}
        sample_index = gen_item.batch["sample_index"].item()
        context_length = gen_item.batch["context_length"].item()
        conversations = []
        global_memory = NO_MEMORY
        final_pass_used = 1
        self.config: MemoryConfig
        chunk_size = self.config.chunk_size
        chunk_parallelism = int(getattr(self.config, "chunk_parallelism_per_sample", 0))
        if chunk_parallelism <= 0:
            chunk_parallelism = max(1, math.ceil(context_length / chunk_size))

        if self._is_trace_sample(sample_index):
            logger.info(
                "[Recurrent][Stage] sample=%d phase=start chunks=%d chunk_parallelism=%d max_passes=%d",
                sample_index,
                math.ceil(context_length / chunk_size),
                chunk_parallelism,
                self.config.max_passes,
            )

        for pass_num in range(self.config.max_passes):
            current_pass = pass_num + 1
            final_pass_used = current_pass
            chunk_memories = []
            all_no_new_info = True
            # Process all chunks in this pass (parallel within pass).
            total_chunks = math.ceil(context_length / chunk_size)
            if total_chunks > self.config.max_chunks:
                logger.warning(
                    f"[AsyncMemoryAgent] {total_chunks=} exceeds {self.config.max_chunks=}, {context_length=}, {pass_num=}"
                )
            with _timer("mt_mics", timing_raw):
                chunk_requests = []
                for step in range(total_chunks):
                    chunk_ids = gen_item.batch["context_ids"][
                        step * chunk_size : (step + 1) * chunk_size
                    ]
                    chunk_ids = chunk_ids[chunk_ids != self.tokenizer.pad_token_id]

                    kwargs = self.sampling_params(gen_item.meta_info)
                    kwargs["max_completion_tokens"] = self.config.max_memorization_length

                    conversation = [
                        {
                            "role": "user",
                            "content": TEMPLATE_CHUNK.format(
                                prompt=gen_item.non_tensor_batch["prompt"],
                                memory=global_memory,
                                chunk=self.tokenizer.decode(chunk_ids),
                            ),
                        }
                    ]
                    chunk_requests.append((step, conversation, kwargs))

            chunk_outputs = []
            for start in range(0, total_chunks, chunk_parallelism):
                req_slice = chunk_requests[start : start + chunk_parallelism]
                if self._is_trace_sample(sample_index):
                    logger.info(
                        "[Recurrent][Stage] sample=%d phase=chunk pass=%d window=%d-%d/%d size=%d",
                        sample_index,
                        current_pass,
                        start + 1,
                        start + len(req_slice),
                        total_chunks,
                        len(req_slice),
                    )
                gen_t0 = time.perf_counter()
                with _timer("mt_async_gen", timing_raw):
                    try:
                        out_slice = await asyncio.gather(
                            *[
                                self.proxy.get_chat_completions(messages=conversation, **kwargs)
                                for _, conversation, kwargs in req_slice
                            ]
                        )
                    except Exception:
                        logger.exception(
                            "[Recurrent][StageError] sample=%d phase=chunk pass=%d window=%d-%d/%d elapsed=%.2fs",
                            sample_index,
                            current_pass,
                            start + 1,
                            start + len(req_slice),
                            total_chunks,
                            time.perf_counter() - gen_t0,
                        )
                        raise
                chunk_outputs.extend(out_slice)

            with _timer("mt_mics", timing_raw):
                # Keep deterministic chunk order for logging/output consistency.
                for (step, conversation, _), (completions, err) in zip(chunk_requests, chunk_outputs):
                    if err:
                        logger.exception(
                            "[Recurrent][StageError] sample=%d phase=chunk pass=%d step=%d/%d",
                            sample_index,
                            current_pass,
                            step + 1,
                            total_chunks,
                            exc_info=err,
                        )
                        raise err

                    choice = completions.choices[0]
                    conversation.append(msg(choice))
                    conversations.append(conversation)
                    content = conversation[-1]["content"]
                    # Detect [NO_NEW_INFO] marker
                    if self.config.no_new_info_marker.lower() in content.lower():
                        chunk_memories.append(None)
                    else:
                        chunk_memories.append(content)
                        all_no_new_info = False

            # Check convergence: if all chunks returned [NO_NEW_INFO], skip merge and stop
            if all_no_new_info:
                if self._is_trace_sample(sample_index):
                    logger.info(
                        "[Recurrent][Stage] sample=%d phase=converged pass=%d marker=%s",
                        sample_index,
                        current_pass,
                        self.config.no_new_info_marker,
                    )
                break

            # Merge step: consolidate only non-None chunk memories
            with _timer("mt_mics", timing_raw):
                memories_text = build_merge_memories(
                    previous_memory=global_memory,
                    new_memories=[m for m in chunk_memories if m is not None],
                )
                kwargs = self.sampling_params(gen_item.meta_info)
                kwargs["max_completion_tokens"] = self.config.max_merge_length

                merge_conversation = [
                    {
                        "role": "user",
                        "content": TEMPLATE_MERGE.format(
                            prompt=gen_item.non_tensor_batch["prompt"],
                            memories=memories_text,
                        ),
                    }
                ]
            if self._is_trace_sample(sample_index):
                logger.info(
                    "[Recurrent][Stage] sample=%d phase=merge pass=%d non_empty_chunks=%d/%d",
                    sample_index,
                    current_pass,
                    sum(m is not None for m in chunk_memories),
                    total_chunks,
                )
            merge_t0 = time.perf_counter()
            with _timer("mt_async_gen", timing_raw):
                try:
                    completions, err = await self.proxy.get_chat_completions(
                        messages=merge_conversation, **kwargs
                    )
                except Exception:
                    logger.exception(
                        "[Recurrent][StageError] sample=%d phase=merge pass=%d elapsed=%.2fs",
                        sample_index,
                        current_pass,
                        time.perf_counter() - merge_t0,
                    )
                    raise
            with _timer("mt_mics", timing_raw):
                if err:
                    logger.exception(
                        "[Recurrent][StageError] sample=%d phase=merge pass=%d",
                        sample_index,
                        current_pass,
                        exc_info=err,
                    )
                    raise err
                choice = completions.choices[0]
                merge_conversation.append(msg(choice))
                conversations.append(merge_conversation)
                global_memory = merge_conversation[-1]["content"].strip()

        # Final turn: generate answer from global memory
        with _timer("mt_mics", timing_raw):
            conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=global_memory,
                    ),
                }
            ]
            kwargs = self.sampling_params(gen_item.meta_info)
            kwargs["max_completion_tokens"] = self.config.max_final_response_length
        if self._is_trace_sample(sample_index):
            logger.info(
                "[Recurrent][Stage] sample=%d phase=final pass_used=%d",
                sample_index,
                final_pass_used,
            )
        final_t0 = time.perf_counter()
        with _timer("mt_async_gen", timing_raw):
            try:
                completions, err = await self.proxy.get_chat_completions(
                    messages=conversation, **kwargs
                )
            except Exception:
                logger.exception(
                    "[Recurrent][StageError] sample=%d phase=final pass=%d elapsed=%.2fs",
                    sample_index,
                    final_pass_used,
                    time.perf_counter() - final_t0,
                )
                raise
        with _timer("mt_mics", timing_raw):
            if err:
                logger.exception(
                    "[Recurrent][StageError] sample=%d phase=final pass=%d",
                    sample_index,
                    final_pass_used,
                    exc_info=err,
                )
                raise err
            choice = completions.choices[0]
            conversation.append(msg(choice))
            conversations.append(conversation)

            sample_index = torch.full(
                (len(conversations),), sample_index, dtype=torch.long
            )
            final_mask = torch.full((len(conversations),), False, dtype=torch.bool)
            final_mask[-1] = True
            pass_used = torch.full(
                (len(conversations),), final_pass_used, dtype=torch.long
            )
        return AsyncOutput(
            conversations,
            sample_index,
            final_mask,
            timing_raw,
            batch={"pass_used": pass_used.numpy()},
        )


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(
    config_cls=MemoryConfig, dataset_cls=AsyncMemoryDataset, agent_cls=AsyncMemoryAgent
)
