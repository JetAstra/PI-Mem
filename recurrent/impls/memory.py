import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override

import verl.utils.torch_functional as verl_F
from recurrent.interface import RAgent, RConfig, RDataset, RRegister
from recurrent.utils import chat_template, unpad
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


@dataclass
class MemoryConfig(RConfig):
    context_key: str
    max_prompt_length: int  #
    chunk_size: int  # size of each context chunk in number of tokens
    max_memorization_length: int  # max number of tokens per-chunk extraction
    max_chunks: int  # max number of chunks to process
    max_final_response_length: int
    max_passes: int = 3  # number of full pass→merge iterations
    max_merge_length: int = 2048  # max output tokens for the merge step
    pass_reward_coef: float = 0.0  # add-on reward: fewer passes -> higher bonus
    # Async memory only: cap concurrent chunk requests per sample in one pass.
    # <=0 means no cap (launch all chunks in parallel, current behavior).
    chunk_parallelism_per_sample: int = 0
    no_new_info_marker: str = (
        "<check>no</check>"  # marker the model outputs when chunk has no new info
    )

    @property
    def max_raw_input_length(self):
        """Chunk phase: prompt + chunk + global_memory (from merge)"""
        return self.max_prompt_length + self.chunk_size + self.max_merge_length

    @property
    def max_raw_merge_input_length(self):
        """Merge phase: prompt + all chunk memories concatenated"""
        return self.max_prompt_length + self.max_chunks * self.max_memorization_length

    @property
    def gen_max_tokens_memorization(self):
        return self.max_memorization_length

    @property
    def gen_max_tokens_merge(self):
        return self.max_merge_length

    @property
    def gen_max_tokens_final_response(self):
        return self.max_final_response_length

    @property
    def gen_pad_to(self):
        return max(
            self.max_memorization_length,
            self.max_merge_length,
            self.max_final_response_length,
        )


class MemoryDataset(RDataset):
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
        logger.info(f"[Check Recurrent Cfg] {recurrent_config}\n")
        if data_config.truncation != 'middle':
            raise ValueError(f'MemoryDataset only support middle truncation, got {data_config.truncation=}')
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
        )  # 将context_ids和attention_mask都处理成max_prompt_length的长度，超过部分会被截断，不足部分会被右侧填充pad_token_id

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

TEMPLATE_FINAL_BOXED = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

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


def build_merge_memories(previous_memory: str | None, new_memories: list[str]) -> str:
    """
    Build the <extracted_information> payload to match parallel_boxed.py exactly.
    """
    parts = []

    if previous_memory and previous_memory.strip() and previous_memory.strip() != NO_MEMORY:
        parts.append(f"{PREVIOUS_MEMORY_LABEL}\n{previous_memory.strip()}")

    for i, mem in enumerate(new_memories, start=1):
        if mem is None:
            continue
        mem = mem.strip()
        if not mem:
            continue
        parts.append(f"{SECTION_LABEL_TEMPLATE.format(idx=i)}\n{mem}")

    return "\n\n".join(parts) if parts else NO_MEMORY


class MemoryAgent(RAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, config: MemoryConfig):
        self.config = config
        self.tokenizer = tokenizer
        self.chat_template = chat_template(tokenizer)

        chunk_template_len = self._encode_user_message(
            TEMPLATE_CHUNK.format(prompt="", memory="", chunk="")
        ).numel()
        merge_template_len = self._encode_user_message(
            TEMPLATE_MERGE.format(prompt="", memories="")
        ).numel()
        final_template_len = self._encode_user_message(
            TEMPLATE_FINAL_BOXED.format(prompt="", memory="")
        ).numel()
        max_label_len = max(
            len(
                tokenizer.encode(
                    f"\n\n{SECTION_LABEL_TEMPLATE.format(idx=i + 1)}\n",
                    add_special_tokens=False,
                )
            )
            for i in range(config.max_chunks)
        )
        previous_label_len = len(
            tokenizer.encode(
                f"{PREVIOUS_MEMORY_LABEL}\n",
                add_special_tokens=False,
            )
        )

        # Compute max input lengths per phase
        self.max_chunk_input_length = config.max_raw_input_length + chunk_template_len
        self.max_merge_input_length = (
            config.max_raw_merge_input_length
            + config.max_chunks * max_label_len
            + previous_label_len
            + merge_template_len
        )
        self.max_final_input_length = (
            config.max_prompt_length
            + config.max_merge_length
            + final_template_len
        )
        self.max_input_length = max(
            self.max_chunk_input_length,
            self.max_merge_input_length,
            self.max_final_input_length,
        )
        logger.info(
            f"\n[RECURRENT] max_input_length: {self.max_input_length} "
            f"(chunk={self.max_chunk_input_length}, merge={self.max_merge_input_length}, "
            f"final={self.max_final_input_length})\n"
        )

    def _encode_user_message(self, content: str) -> torch.LongTensor:
        return torch.tensor(
            self.tokenizer.encode(
                self.chat_template.format(message=content),
                add_special_tokens=False,
            ),
            dtype=torch.long,
        )

    def _decode_text(self, value) -> str:
        if value is None:
            return NO_MEMORY
        if isinstance(value, torch.Tensor):
            return self.tokenizer.decode(value, skip_special_tokens=True).strip()
        return str(value).strip()

    @override
    def start(self, gen_batch: DataProto, timing_raw: dict):
        self.gen_batch = gen_batch
        self.step = 0
        self.final_mask_list = []
        self.sample_index_list = []

        self.ctx_length = gen_batch.batch["context_length"]
        self.bsz = len(self.ctx_length)

        # Multi-pass state
        self.phase = "chunk"  # "chunk" | "merge" | "final"
        self.pass_num = 0
        self.chunk_step = 0
        self.global_memory = np.full(self.bsz, None, dtype=object)
        self.chunk_memories = []  # list of np.ndarray(object), one per chunk step
        self.is_done = False

        # Per-sample convergence state
        self.converged = np.full(self.bsz, False, dtype=bool)
        # Per-sample pass index used by reward shaping on final answers.
        # Record the first pass when a sample converges; unresolved samples
        # will be filled with the pass index when the batch enters final phase.
        self.sample_pass_used = np.full(self.bsz, -1, dtype=np.int32)

    def _is_no_new_info(self, response_tokens: torch.Tensor) -> bool:
        """Check if the model's response contains <check>no</check>."""
        text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        return self.config.no_new_info_marker.lower() in text.lower()

    def _check_pass_convergence(self):
        """After all chunks in a pass, mark samples as converged if all their chunks were [NO_NEW_INFO]."""
        for i in range(self.bsz):
            if self.converged[i]:
                continue
            has_new_info = any(cm[i] is not None for cm in self.chunk_memories)
            if not has_new_info:
                self.converged[i] = True
                if self.sample_pass_used[i] < 0:
                    self.sample_pass_used[i] = self.pass_num + 1
                logger.info(
                    f"[CONVERGENCE] Sample {i} converged at pass {self.pass_num} "
                    f"(all chunks returned {self.config.no_new_info_marker})"
                )

    def _finalize_sample_pass_used(self, final_pass: int):
        final_pass = max(int(final_pass), 1)
        unset_mask = self.sample_pass_used < 0
        if np.any(unset_mask):
            self.sample_pass_used[unset_mask] = final_pass

    @override
    def action(self) -> Tuple[List[torch.Tensor], dict]:
        if self.phase == "chunk":
            return self._chunk_action()
        elif self.phase == "merge":
            return self._merge_action()
        else:  # "final"
            return self._final_action()

    def _chunk_action(self) -> Tuple[List[torch.Tensor], dict]:
        # Active = has remaining chunks AND not yet converged
        active_mask = (
            self.ctx_length > self.chunk_step * self.config.chunk_size
        ) & torch.tensor(~self.converged)
        self.active_mask = active_mask

        if active_mask.sum().item() == 0:
            # All chunks exhausted or all converged in this pass
            self._check_pass_convergence()
            if self.converged.all():
                self.pass_num += 1
                self._finalize_sample_pass_used(self.pass_num)
                self.phase = "final"
                logger.info(
                    f"[CONVERGENCE] All samples converged at pass {self.pass_num - 1}, skipping merge"
                )
                return self._final_action()
            else:
                self.phase = "merge"
                return self._merge_action()

        gen_batch = self.gen_batch
        prompt_i = gen_batch.non_tensor_batch["prompt"][active_mask]
        chunk_i = gen_batch.batch["context_ids"][
            active_mask,
            self.config.chunk_size
            * self.chunk_step : self.config.chunk_size
            * (self.chunk_step + 1),
        ]
        memory_i = self.global_memory[active_mask]

        self.messages = [
            self._encode_user_message(
                TEMPLATE_CHUNK.format(
                    prompt=prompt,
                    memory=self._decode_text(memory),
                    chunk=self.tokenizer.decode(
                        chunk[chunk != self.tokenizer.pad_token_id]
                    ),
                )
            )
            for prompt, memory, chunk in zip(prompt_i, memory_i, chunk_i)
        ]

        sample_index = torch.arange(self.bsz, dtype=torch.long)[active_mask]
        final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)
        self.meta_info = {
            "input_pad_to": self.max_input_length,
            "pad_to": self.config.gen_pad_to,
            "generation_kwargs": {
                "max_tokens": self.config.gen_max_tokens_memorization,
                "n": 1,
            },
        }
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        logger.info(
            f"CHUNK ACTION: pass={self.pass_num}, chunk_step={self.chunk_step}, "
            f"active={active_mask.sum().item()}/{self.bsz}, "
            f"converged={self.converged.sum()}/{self.bsz}"
        )
        return self.messages, self.meta_info

    def _merge_action(self) -> Tuple[List[torch.Tensor], dict]:
        gen_batch = self.gen_batch
        non_converged = ~self.converged
        self.merge_mask = torch.tensor(non_converged)

        self.messages = []
        for i in range(self.bsz):
            if self.converged[i]:
                continue
            new_memories = [
                self._decode_text(chunk_mem_array[i])
                for chunk_mem_array in self.chunk_memories
                if chunk_mem_array[i] is not None
            ]
            memories = build_merge_memories(
                previous_memory=(
                    self._decode_text(self.global_memory[i])
                    if self.global_memory[i] is not None
                    else None
                ),
                new_memories=new_memories,
            )
            msg = self._encode_user_message(
                TEMPLATE_MERGE.format(
                    prompt=gen_batch.non_tensor_batch["prompt"][i],
                    memories=memories,
                )
            )
            self.messages.append(msg)

        sample_index = torch.arange(self.bsz, dtype=torch.long)[self.merge_mask]
        final_mask = torch.full(sample_index.shape, False, dtype=torch.bool)
        self.meta_info = {
            "input_pad_to": self.max_input_length,
            "pad_to": self.config.gen_pad_to,
            "generation_kwargs": {
                "max_tokens": self.config.gen_max_tokens_merge,
                "n": 1,
            },
        }
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        logger.info(
            f"MERGE ACTION: pass={self.pass_num}, "
            f"merging={non_converged.sum()}/{self.bsz} samples"
        )
        return self.messages, self.meta_info

    def _final_action(self) -> Tuple[List[torch.Tensor], dict]:
        gen_batch = self.gen_batch
        self.messages = [
            self._encode_user_message(
                TEMPLATE_FINAL_BOXED.format(
                    prompt=prompt,
                    memory=self._decode_text(memory),
                )
            )
            for prompt, memory in zip(
                gen_batch.non_tensor_batch["prompt"], self.global_memory
            )
        ]
        sample_index = torch.arange(self.bsz, dtype=torch.long)
        final_mask = torch.full(sample_index.shape, True, dtype=torch.bool)
        self.meta_info = {
            "input_pad_to": self.max_input_length,
            "pad_to": self.config.gen_pad_to,
            "generation_kwargs": {
                "max_tokens": self.config.gen_max_tokens_final_response,
                "n": 1,
            },
        }
        self.final_mask_list.append(final_mask)
        self.sample_index_list.append(sample_index)
        logger.info(
            f"FINAL ACTION: converged={self.converged.sum()}/{self.bsz} samples"
        )
        return self.messages, self.meta_info

    @override
    def update(self, gen_output: DataProto) -> DataProto:
        # Track pass count per generated turn so trainer can shape reward on final answers.
        if self.phase == "final":
            self._finalize_sample_pass_used(self.pass_num)
            gen_output.batch["pass_used"] = torch.as_tensor(
                self.sample_pass_used, dtype=torch.long
            )
        else:
            current_pass = self.pass_num + 1
            gen_output.batch["pass_used"] = torch.full(
                (len(gen_output),), current_pass, dtype=torch.long
            )

        if self.phase == "chunk":
            # Save per-chunk memory, detect [NO_NEW_INFO]
            chunk_mem = np.full(self.bsz, None, dtype=object)
            responses = unpad(
                self.tokenizer, gen_output.batch["responses"], remove_eos=True
            )
            active_indices = torch.where(self.active_mask)[0]
            for resp_idx, sample_idx_tensor in enumerate(active_indices):
                sample_idx = sample_idx_tensor.item()
                if self._is_no_new_info(responses[resp_idx]):
                    chunk_mem[sample_idx] = None  # mark as no new info
                else:
                    chunk_mem[sample_idx] = responses[resp_idx]
            self.chunk_memories.append(chunk_mem)
            self.chunk_step += 1

            # Check if next chunk step has any active (non-converged + has chunks) samples
            next_active = (
                self.ctx_length > self.chunk_step * self.config.chunk_size
            ) & torch.tensor(~self.converged)
            if next_active.sum().item() == 0:
                self._check_pass_convergence()
                if self.converged.all():
                    self.pass_num += 1
                    self._finalize_sample_pass_used(self.pass_num)
                    self.phase = "final"
                else:
                    self.phase = "merge"

        elif self.phase == "merge":
            # Store merged result only for non-converged samples
            new_memories = unpad(
                self.tokenizer, gen_output.batch["responses"], remove_eos=True
            )
            self.global_memory[self.merge_mask] = new_memories
            self.pass_num += 1
            if self.pass_num >= self.config.max_passes or self.converged.all():
                self._finalize_sample_pass_used(self.pass_num)
                self.phase = "final"
            else:
                # Reset for next pass
                self.chunk_step = 0
                self.chunk_memories = []
                self.phase = "chunk"

        elif self.phase == "final":
            self.is_done = True

        self.log_step(gen_output)
        self.step += 1
        return gen_output

    @override
    def done(self):
        return self.is_done

    @override
    def end(self):
        del self.gen_batch
        del self.ctx_length
        del self.meta_info
        del self.global_memory
        del self.chunk_memories
        del self.messages
        del self.converged
        del self.sample_pass_used
        sample_index = torch.cat(self.sample_index_list)
        final_mask = torch.cat(self.final_mask_list)
        del self.final_mask_list
        del self.sample_index_list
        return final_mask, sample_index

    def log_step(self, gen_output):
        """Log multi-turn conversation details."""

        def clip_long_string(string, max_length=2000):
            if not len(string) > max_length:
                return string
            return (
                string[: max_length // 2]
                + "\n\n...(ignored)\n\n"
                + string[-max_length // 2 :]
            )

        step_label = f"{self.phase.upper()} pass={self.pass_num} step={self.step}"
        logger.info(f"\n{'=' * 30}[RECURRENT] {step_label}{'=' * 30}")

        # Log first sample if available
        show_idx = 0
        if self.phase == "chunk" and not self.active_mask[0]:
            logger.info("MESSAGE and RESPONSE is empty since sample 0 is not active.")
            return

        if show_idx < len(self.messages):
            pass
            # decoded_message = self.tokenizer.decode(self.messages[show_idx])
            # rsp0 = gen_output.batch["responses"][show_idx]
            # decoded_response = self.tokenizer.decode(
            #     rsp0[rsp0 != self.tokenizer.pad_token_id]
            # )
            # logger.info(f"[MESSAGE] {clip_long_string(decoded_message)}")
            # logger.info(f"{' ' * 10}{'-' * 20}prompt end{'-' * 20}{' ' * 10}")
            # logger.info(f"[RESPONSE] {decoded_response}")
            # logger.info(f"{' ' * 10}{'-' * 20}response end{'-' * 20}{' ' * 10}")


# Important, we will import `REGISTER` from this file to get all registered classes.
# specified by recurrent.path / recurrent.name(defaults to REGISTER)
REGISTER = RRegister(
    config_cls=MemoryConfig, dataset_cls=MemoryDataset, agent_cls=MemoryAgent
)
