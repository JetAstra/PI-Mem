import asyncio
import logging
import math
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
from recurrent.async_utils import ChatCompletionProxy


from recurrent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from recurrent.utils import log_step, msg
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
            raise ValueError("MemoryDataset only support center truncation")
        chunk_size = recurrent_config.chunk_size
        old_max_chunks = recurrent_config.max_chunks
        old_max_prompt_length = data_config.max_prompt_length

        # If both are set, trust prompt-length budget and align max_chunks by ceil.
        if old_max_chunks is not None and old_max_prompt_length is not None:
            new_max_chunks = math.ceil(old_max_prompt_length / chunk_size)
            recurrent_config.max_chunks = new_max_chunks
            data_config.max_prompt_length = new_max_chunks * chunk_size
            logger.info(
                "[MemoryDataset] recompute by max_prompt_length/chunk_size: "
                f"max_chunks {old_max_chunks} -> {new_max_chunks}, "
                f"max_prompt_length {old_max_prompt_length} -> {data_config.max_prompt_length} "
                f"(chunk_size={recurrent_config.chunk_size})"
            )
        elif old_max_chunks is None and old_max_prompt_length is not None:
            recurrent_config.max_chunks = math.ceil(old_max_prompt_length / chunk_size)
            data_config.max_prompt_length = recurrent_config.max_chunks * chunk_size
            logger.info(
                "[MemoryDataset] infer max_chunks from max_prompt_length: "
                f"max_chunks None -> {recurrent_config.max_chunks}, "
                f"max_prompt_length {old_max_prompt_length} -> {data_config.max_prompt_length} "
                f"(chunk_size={recurrent_config.chunk_size})"
            )
        elif old_max_chunks is not None:
            data_config.max_prompt_length = old_max_chunks * chunk_size
            logger.info(
                "[MemoryDataset] infer max_prompt_length from max_chunks: "
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
        context = row_dict.pop(self.context_key)

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
        row_dict["prompt"] = chat[0]["content"]
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
        global_memory = None
        self.config: MemoryConfig
        chunk_size = self.config.chunk_size

        for pass_num in range(self.config.max_passes):
            chunk_memories = []
            all_no_new_info = True
            chunk_count = math.ceil(context_length / chunk_size)
            assert (
                chunk_count <= self.config.max_chunks
            ), f"{chunk_count=} exceeds {self.config.max_chunks=}, {context_length=}"

            async def rollout_one_chunk(step: int):
                with _timer("mt_mics", timing_raw):
                    chunk_ids = gen_item.batch["context_ids"][
                        step * chunk_size : (step + 1) * chunk_size
                    ]
                    kwargs = self.sampling_params(gen_item.meta_info)
                    if sample_index == 0 and step == 0:
                        logger.info(f"generate_sequences sampling params: {kwargs}")
                    kwargs["max_completion_tokens"] = (
                        self.config.max_memorization_length
                    )
                    conversation = [
                        {
                            "role": "user",
                            "content": TEMPLATE_CHUNK.format(
                                prompt=gen_item.non_tensor_batch["prompt"],
                                memory=(
                                    global_memory
                                    if global_memory
                                    else "No previous memory"
                                ),
                                chunk=self.tokenizer.decode(
                                    chunk_ids, skip_special_tokens=True
                                ),
                            ),
                        }
                    ]
                completions, err = await self.proxy.get_chat_completions(
                    messages=conversation, **kwargs
                )
                return step, conversation, completions, err

            with _timer("mt_async_gen", timing_raw):
                chunk_results = await asyncio.gather(
                    *[rollout_one_chunk(step) for step in range(chunk_count)]
                )
            with _timer("mt_mics", timing_raw):
                for step, conversation, completions, err in chunk_results:
                    if err:
                        raise err
                    choice = completions.choices[0]
                    conversation.append(msg(choice))
                    conversations.append(conversation)
                    content = conversation[-1]["content"]
                    # Detect [NO_NEW_INFO] marker
                    if self.config.no_new_info_marker in content:
                        chunk_memories.append(None)
                    else:
                        chunk_memories.append(content)
                        all_no_new_info = False
                    if sample_index == 0:
                        log_step(logger, f"pass{pass_num}_chunk{step}", conversation)

            # Check convergence: if all chunks returned [NO_NEW_INFO], skip merge and stop
            if all_no_new_info:
                if sample_index == 0:
                    logger.info(
                        f"[CONVERGENCE] Sample converged at pass {pass_num} "
                        f"(all chunks returned {self.config.no_new_info_marker})"
                    )
                break

            # Merge step: consolidate only non-None chunk memories
            with _timer("mt_mics", timing_raw):
                valid_memories = [
                    (i, m) for i, m in enumerate(chunk_memories) if m is not None
                ]
                memories_text = "\n".join(
                    f"[Section {i + 1}]:\n{m}" for i, m in valid_memories
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
            with _timer("mt_async_gen", timing_raw):
                completions, err = await self.proxy.get_chat_completions(
                    messages=merge_conversation, **kwargs
                )
            with _timer("mt_mics", timing_raw):
                if err:
                    raise err
                choice = completions.choices[0]
                merge_conversation.append(msg(choice))
                conversations.append(merge_conversation)
                global_memory = merge_conversation[-1]["content"]
                if sample_index == 0:
                    log_step(logger, f"pass{pass_num}_merge", merge_conversation)

        # Final turn: generate answer from global memory
        with _timer("mt_mics", timing_raw):
            conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=global_memory if global_memory else "No previous memory",
                    ),
                }
            ]
            kwargs = self.sampling_params(gen_item.meta_info)
            kwargs["max_completion_tokens"] = self.config.max_final_response_length
        with _timer("mt_async_gen", timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages=conversation, **kwargs
            )
        with _timer("mt_mics", timing_raw):
            if err:
                raise err
            choice = completions.choices[0]
            conversation.append(msg(choice))
            conversations.append(conversation)
            if sample_index == 0:
                log_step(logger, "final", conversation)

            sample_index = torch.full(
                (len(conversations),), sample_index, dtype=torch.long
            )
            final_mask = torch.full((len(conversations),), False, dtype=torch.bool)
            final_mask[-1] = True
        return AsyncOutput(conversations, sample_index, final_mask, timing_raw)


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(
    config_cls=MemoryConfig, dataset_cls=AsyncMemoryDataset, agent_cls=AsyncMemoryAgent
)
