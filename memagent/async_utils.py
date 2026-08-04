# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


class ChatCompletionProxy:
    """
    Adapter from the old OpenAI chat-completions interface used by memagent to
    the current verl LLMServerClient token-in/token-out API.
    """

    def __init__(
        self,
        llm_client,
        tokenizer,
        model_name: str = "memagent",
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.llm_client = llm_client
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}

    async def get_chat_completions(
        self,
        model=None,
        **chat_complete_request,
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Submit a chat-completion-like request and return the fields consumed by
        existing memagent agents: choices[0].message, finish_reason, stop_reason.
        """
        completions, exception = None, None
        start_time = time.monotonic()
        try:
            extra_headers = chat_complete_request.pop("extra_headers", {}) or {}
            request_id = extra_headers.get("x-request-id", None)
            if request_id and request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
            if not request_id:
                request_id = uuid4().hex
            should_log = os.environ.get("MEMAGENT_LOG_ALL_REQUESTS", "0") == "1"
            if request_id.startswith("memagent-s"):
                try:
                    sample_idx = int(request_id.split("-", 2)[1][1:])
                    should_log = should_log or sample_idx < 8 or sample_idx % 128 == 0
                except Exception:
                    should_log = True

            messages = chat_complete_request.pop("messages")
            prompt_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()

            sampling_params = {}
            if "max_completion_tokens" in chat_complete_request:
                sampling_params["max_tokens"] = chat_complete_request.pop("max_completion_tokens")
            if "max_tokens" in chat_complete_request:
                sampling_params["max_tokens"] = chat_complete_request.pop("max_tokens")
            if "max_new_tokens" in chat_complete_request:
                sampling_params["max_new_tokens"] = chat_complete_request.pop("max_new_tokens")
            for key in ("temperature", "top_p", "top_k", "min_p", "stop", "logprobs"):
                if key in chat_complete_request:
                    sampling_params[key] = chat_complete_request.pop(key)

            if chat_complete_request.get("best_of", 1) != 1:
                raise NotImplementedError("best_of > 1 is not supported by memagent ChatCompletionProxy")
            n = chat_complete_request.get("n", 1)
            if n != 1:
                raise NotImplementedError("n > 1 is not supported by memagent ChatCompletionProxy")

            if should_log:
                logger.info(
                    "[MemAgent][proxy] generate start request_id=%s prompt_tokens=%d max_tokens=%s "
                    "temperature=%s top_p=%s",
                    request_id,
                    len(prompt_ids),
                    sampling_params.get("max_tokens", sampling_params.get("max_new_tokens")),
                    sampling_params.get("temperature"),
                    sampling_params.get("top_p"),
                )
            output = await self.llm_client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            if should_log:
                logger.info(
                    "[MemAgent][proxy] generate done request_id=%s output_tokens=%d stop_reason=%s elapsed=%.1fs",
                    request_id,
                    len(output.token_ids),
                    output.stop_reason,
                    time.monotonic() - start_time,
                )
            content = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            finish_reason = "stop" if output.stop_reason == "completed" else output.stop_reason
            choice = SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content),
                finish_reason=finish_reason,
                stop_reason=None,
            )
            completions = SimpleNamespace(
                id=f"chatcmpl-{request_id}",
                model=model or self.model_name,
                choices=[choice],
            )
        except Exception as e:
            # Let user handle the exception
            exception = e
            logger.exception(
                "[MemAgent][proxy] generate failed request_id=%s elapsed=%.1fs",
                locals().get("request_id", None),
                time.monotonic() - start_time,
            )

        return completions, exception


def run_coroutine(coro):
    return asyncio.run(coro)
