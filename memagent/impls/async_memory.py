import logging
import math
import time
from typing import List, Optional, Tuple, Union, Dict
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
from memagent.async_utils import ChatCompletionProxy


from memagent.interface import AsyncRAgent, RConfig, RDataset, RRegister, AsyncOutput
from memagent.utils import log_step, msg
from verl.protocol import DataProtoItem
from memagent.generation_manager import _timer
import verl.utils.torch_functional as verl_F


logger = logging.getLogger(__file__)
logger.setLevel('INFO')


from memagent.impls.memory import MemoryConfig, TEMPLATE, TEMPLATE_FINAL_BOXED

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
            raise ValueError(f"AsyncMemoryDataset only support middle truncation, got {data_config.truncation=}")
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

        model_inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)

        context_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        context_ids, attention_mask = verl_F.postprocess_data(
            input_ids=context_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id, # pyright: ignore
            left_pad=False,
            truncation=self.truncation,
        )

        row_dict["context_ids"] = context_ids[0]
        lengths = attention_mask.sum(dim=-1)
        row_dict["context_length"] = lengths[0]
        row_dict["prompt"] = chat[0]["content"].strip()
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
    def __init__(self, proxy: ChatCompletionProxy, tokenizer: PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        super().__init__(proxy, tokenizer, config, rollout_config)

    def start(self, gen_batch, timing_raw: dict):
        self._debug_progress = {}

    def _set_debug_progress(self, sample_index: int, phase: str, step: int, **kwargs):
        state = {
            "phase": phase,
            "step": step,
            "updated_at": time.monotonic(),
        }
        state.update(kwargs)
        self._debug_progress[sample_index] = state

    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        tensor keys in output:
        - standard verl: "prompts", "responses", "input_ids", "attention_mask", "position_ids"
        - recurrent rl: "sample_index", "final_mask"
        > input_ids = torch.cat([prompts, responses], dim=1)
        """
        timing_raw = {}
        sample_index = gen_item.batch['sample_index'].item()
        context_length = gen_item.batch["context_length"].item()
        step = 0
        conversations = []
        memory = None
        self._set_debug_progress(
            sample_index,
            "start",
            step,
            context_length=context_length,
        )
        while True:
            with _timer('mt_mics', timing_raw):
                if step * self.config.chunk_size >= context_length:
                    break
                assert step < self.config.max_chunks, f"{step=} exceeds {self.config.max_chunks=}, {context_length=}"
                chunk_ids = gen_item.batch["context_ids"]\
                    [step * self.config.chunk_size: (step + 1) * self.config.chunk_size]
                chunk_ids = chunk_ids[chunk_ids != self.tokenizer.pad_token_id]

                kwargs = self.sampling_params(gen_item.meta_info)
                if sample_index == 0:
                    logger.info(f"generate_sequences sampling params: {kwargs}")
                self.config : MemoryConfig
                kwargs["max_completion_tokens"] = self.config.max_memorization_length

                conversation = [
                    {"role": "user", "content": TEMPLATE.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=memory if memory else "No previous memory",
                        chunk=self.tokenizer.decode(chunk_ids, skip_special_tokens=True),
                    )}
                ]
                request_id = f"memagent-s{sample_index}-memory-{step}-{uuid4().hex[:8]}"
                prompt_chars = len(conversation[0]["content"])
                self._set_debug_progress(
                    sample_index,
                    "memory_request",
                    step,
                    context_length=context_length,
                    chunk_tokens=int(chunk_ids.numel()),
                    prompt_chars=prompt_chars,
                    request_id=request_id,
                    max_tokens=kwargs["max_completion_tokens"],
                )
                if sample_index < 8 or sample_index % 128 == 0:
                    logger.info(
                        "[MemAgent][sample] request sample=%d phase=memory step=%d chunk_tokens=%d "
                        "prompt_chars=%d max_tokens=%d request_id=%s",
                        sample_index,
                        step,
                        int(chunk_ids.numel()),
                        prompt_chars,
                        kwargs["max_completion_tokens"],
                        request_id,
                    )
            with _timer('mt_async_gen', timing_raw):
                completions, err = await self.proxy.get_chat_completions(
                    messages=conversation,
                    extra_headers={"x-request-id": f"chatcmpl-{request_id}"},
                    **kwargs
                )
                if err:
                    self._set_debug_progress(
                        sample_index,
                        "failed",
                        step,
                        context_length=context_length,
                        prompt_chars=prompt_chars,
                        request_id=request_id,
                    )
                    raise err
            with _timer('mt_mics', timing_raw):
                choice = completions.choices[0]
                conversation.append(msg(choice))
                conversations.append(conversation)
                memory = conversation[-1]["content"]
                self._set_debug_progress(
                    sample_index,
                    "memory_response",
                    step,
                    context_length=context_length,
                    prompt_chars=prompt_chars,
                    request_id=request_id,
                    response_chars=len(memory),
                )
                if sample_index < 8 or sample_index % 128 == 0:
                    logger.info(
                        "[MemAgent][sample] response sample=%d phase=memory step=%d response_chars=%d",
                        sample_index,
                        step,
                        len(memory),
                    )
                if sample_index == 0:
                    log_step(logger, step, conversation)
            step += 1

        # final turn
        with _timer('mt_mics', timing_raw):
            conversation = [
                {
                    "role": "user",
                    "content": TEMPLATE_FINAL_BOXED.format(
                        prompt=gen_item.non_tensor_batch["prompt"],
                        memory=memory if memory else "No previous memory",
                    ),
                }
            ]
            kwargs = self.sampling_params(gen_item.meta_info)
            kwargs["max_completion_tokens"] = self.config.max_final_response_length
            request_id = f"memagent-s{sample_index}-final-{step}-{uuid4().hex[:8]}"
            prompt_chars = len(conversation[0]["content"])
            self._set_debug_progress(
                sample_index,
                "final_request",
                step,
                context_length=context_length,
                prompt_chars=prompt_chars,
                request_id=request_id,
                max_tokens=kwargs["max_completion_tokens"],
            )
            if sample_index < 8 or sample_index % 128 == 0:
                logger.info(
                    "[MemAgent][sample] request sample=%d phase=final step=%d prompt_chars=%d "
                    "max_tokens=%d request_id=%s",
                    sample_index,
                    step,
                    prompt_chars,
                    kwargs["max_completion_tokens"],
                    request_id,
                )
        with _timer('mt_async_gen', timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages=conversation,
                extra_headers={"x-request-id": f"chatcmpl-{request_id}"},
                **kwargs
            )
        with _timer('mt_mics', timing_raw):
            if err:
                self._set_debug_progress(
                    sample_index,
                    "failed",
                    step,
                    context_length=context_length,
                    prompt_chars=prompt_chars,
                    request_id=request_id,
                )
                raise err
            choice = completions.choices[0]
            conversation.append(msg(choice))
            conversations.append(conversation)
            self._set_debug_progress(
                sample_index,
                "final_response",
                step,
                context_length=context_length,
                prompt_chars=prompt_chars,
                request_id=request_id,
                response_chars=len(conversation[-1]["content"]),
            )
            if sample_index < 8 or sample_index % 128 == 0:
                logger.info(
                    "[MemAgent][sample] response sample=%d phase=final step=%d response_chars=%d",
                    sample_index,
                    step,
                    len(conversation[-1]["content"]),
                )
            if sample_index == 0:
                log_step(logger, step, conversation)

            sample_index = torch.full((len(conversations),), sample_index, dtype=torch.long)
            final_mask = torch.full((len(conversations),), False, dtype=torch.bool)
            final_mask[-1] = True
        self._set_debug_progress(
            int(sample_index[0].item()),
            "done",
            step,
            context_length=context_length,
            turns=len(conversations),
        )
        return AsyncOutput(conversations, sample_index, final_mask, timing_raw)


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=AsyncMemoryDataset, agent_cls=AsyncMemoryAgent)
