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
from typing import Any, Dict, Optional, List
import json
import logging
import shlex

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
try:
    # vLLM>=0.13 (including 0.18.x)
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
        CompletionResponse,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.models.serving import OpenAIServingModels
    from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.executor.ray_executor import RayDistributedExecutor
    PromptAdapterPath = Any
    _VLLM_OPENAI_NEW_LAYOUT = True
except ModuleNotFoundError:
    # vLLM<=0.12 fallback
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        CompletionRequest,
        CompletionResponse,
        ErrorResponse,
    )
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath, LoRAModulePath, PromptAdapterPath
    from vllm.executor.ray_distributed_executor import RayDistributedExecutor
    from vllm.utils import FlexibleArgumentParser
    _VLLM_OPENAI_NEW_LAYOUT = False
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")
import loguru

app = FastAPI()

@serve.deployment(
    num_replicas=8,
    max_ongoing_requests=256,
    logging_config=dict(log_level="WARNING"),
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.serving_models = None
        self.openai_serving_render = None
        self.openai_serving_chat = None
        self.openai_serving_completion = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self._effective_config_printed = False

        if engine_args.model.startswith("hdfs:"):
            from hdfs import copy_local_path_from_hdfs
            print(f'start download from {engine_args.model}')
            import os
            local_model_path = copy_local_path_from_hdfs(src=engine_args.model, cache_dir=os.path.expanduser("~/.cache/verl/rlhf"))
            print('finish download')
        else:
            print(f"load from local dir {engine_args.model}")
            local_model_path = engine_args.model
        from pathlib import Path
        if Path(local_model_path).is_dir():
            self.model_name = Path(local_model_path).name
        else:
            self.model_name = local_model_path
        print(self.model_name)
        engine_args.disable_log_requests=True
        engine_args.model = local_model_path
        engine_args.tokenizer = local_model_path
        engine_args.distributed_executor_backend = RayDistributedExecutor
        import os
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # https://github.com/vllm-project/vllm/issues/8402
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def get_engine_model_config(self):
        if hasattr(self.engine, "get_model_config"):
            return await self.engine.get_model_config()
        return getattr(self.engine, "model_config", None)

    async def print_effective_config_once(self, model_config=None):
        if self._effective_config_printed:
            return
        if model_config is None:
            model_config = await self.get_engine_model_config()
        hf_config = getattr(model_config, "hf_config", None)
        text_config = None
        if hf_config is not None:
            if hasattr(hf_config, "get_text_config"):
                text_config = hf_config.get_text_config()
            else:
                text_config = getattr(hf_config, "text_config", None)
        payload = {
            "model_config.max_model_len": getattr(model_config, "max_model_len", None),
            "engine_args.hf_overrides": getattr(self.engine_args, "hf_overrides", None),
            "text_config.max_position_embeddings": getattr(
                text_config, "max_position_embeddings", None
            ),
            "text_config.rope_parameters": getattr(
                text_config, "rope_parameters", None
            ),
            "text_config.rope_scaling": getattr(text_config, "rope_scaling", None),
        }
        loguru.logger.debug(
            f"========== vLLM effective config ==========\n"
            f"{json.dumps(payload, ensure_ascii=False, default=str, indent=2)}\n"
            f"==========================================="
        )
        self._effective_config_printed = True

    async def get_models(self):
        if not self.serving_models:
            model_config = await self.get_engine_model_config()
            await self.print_effective_config_once(model_config)
            kwargs = dict(
                engine_client=self.engine,
                base_model_paths=[
                    BaseModelPath(
                        name=self.model_name, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
            )
            if not _VLLM_OPENAI_NEW_LAYOUT:
                kwargs["model_config"] = model_config
                kwargs["prompt_adapters"] = self.prompt_adapters
            self.serving_models = OpenAIServingModels(**kwargs)
        return self.serving_models

    async def get_openai_serving_render(self, models=None):
        if not _VLLM_OPENAI_NEW_LAYOUT:
            return None
        if not self.openai_serving_render:
            from vllm.entrypoints.serve.render.serving import OpenAIServingRender

            if models is None:
                models = await self.get_models()
            self.openai_serving_render = OpenAIServingRender(
                model_config=models.model_config,
                renderer=models.renderer,
                io_processor=models.io_processor,
                model_registry=models,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
                trust_request_chat_template=False,
                enable_auto_tools=False,
                exclude_tools_when_tool_choice_none=False,
                tool_parser=None,
                default_chat_template_kwargs=None,
            )
        return self.openai_serving_render
    
    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            if _VLLM_OPENAI_NEW_LAYOUT:
                models = await self.get_models()
                openai_serving_render = await self.get_openai_serving_render(models)

                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    models=models,
                    response_role=self.response_role,
                    openai_serving_render=openai_serving_render,
                    request_logger=self.request_logger,
                    chat_template=self.chat_template,
                    chat_template_content_format="auto",
                    # return_tokens_as_token_ids: bool = False,
                    # reasoning_parser: str = "",
                    # enable_auto_tools: bool = False,
                    # tool_parser: str | None = None,
                    # enable_prompt_tokens_details: bool = False
                )
            else:
                model_config = await self.engine.get_model_config()
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    models=await self.get_models(),
                    response_role=self.response_role,
                    request_logger=self.request_logger,
                    chat_template=self.chat_template,
                    chat_template_content_format=None,
                    # return_tokens_as_token_ids: bool = False,
                    # enable_reasoning: bool = False,
                    # reasoning_parser: str | None = None,
                    # enable_auto_tools: bool = False,
                    # tool_parser: str | None = None,
                    # enable_prompt_tokens_details: bool = False
                )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    @app.post("/v1/completions")
    async def create_completion(
        self, request: CompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible text completion endpoint for prompt-string callers."""
        if not self.openai_serving_completion:
            if _VLLM_OPENAI_NEW_LAYOUT:
                models = await self.get_models()
                openai_serving_render = await self.get_openai_serving_render(models)
                self.openai_serving_completion = OpenAIServingCompletion(
                    self.engine,
                    models=models,
                    openai_serving_render=openai_serving_render,
                    request_logger=self.request_logger,
                )
            else:
                model_config = await self.engine.get_model_config()
                self.openai_serving_completion = OpenAIServingCompletion(
                    self.engine,
                    model_config,
                    models=await self.get_models(),
                    request_logger=self.request_logger,
                )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
            return JSONResponse(content=generator.model_dump())
        
    @app.get("/v1/models")
    async def show_available_models(self, raw_request: Request):
        model_config= await self.get_models()
        models = await model_config.show_available_models()
        return JSONResponse(content=models.model_dump())

def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    boolean_flags = {
        "enforce-eager",
        "disable-custom-all-reduce",
    }
    for key, value in cli_args.items():
        if key in boolean_flags:
            if str(value).lower() in {"1", "true", "yes", "on"}:
                arg_strings.append(f"--{key}")
            continue
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    # Deployment replica will also use GPU for AsyncLLMEngine.
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors, 

    print(f"{tp=}, {parsed_args=}, {engine_args=}")
    print("========================================")
    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    import os
    ray.init(dashboard_port=int(os.getenv("DASH_PORT", "8265")))
    available_gpus = ray.available_resources()["GPU"]
    return VLLMDeployment.options(
        num_replicas=available_gpus // tp,
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        getattr(parsed_args, "prompt_adapters", None),
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )

if __name__ == "__main__":
    # a quicker way
    import os
    import argparse
    # Hard-coded scheduler knobs for better utilization under Ray Serve.
    HARDCODED_MAX_NUM_BATCHED_TOKENS = 131072
    # HARDCODED_MAX_NUM_SEQS = 128
    pwd = os.path.dirname(os.path.abspath(__file__))
    file = os.path.splitext(os.path.basename(__file__))[0]
    parser = argparse.ArgumentParser(description="Ray Serve + vLLM deployment. Usage: python llm070.py --model Qwen/Qwen2.5-7B-Instruct --tp 2")
    parser.add_argument('--model', type=str, required=True,help='model name or path, e.g. Qwen/Qwen2.5-7B-Instruct or /mnt/hdfs/model/MemoryAgent-14B')
    parser.add_argument('--tp', type=int, default=1, help='tensor parallel size')
    parser.add_argument('--port', type=int, default=int(os.getenv("SERVE_PORT", "8000")), help='OpenAI API HTTP port')
    parser.add_argument('--dash-port', type=int, default=int(os.getenv("DASH_PORT", "8265")), help='Ray dashboard port')
    parser.add_argument('--max-model-len', type=int, default=None, help='Forward to vLLM --max-model-len')
    parser.add_argument('--gpu-memory-utilization', type=float, default=None, help='Forward to vLLM --gpu-memory-utilization')
    parser.add_argument('--enforce-eager', action='store_true', help='Forward to vLLM --enforce-eager')
    parser.add_argument('--disable-custom-all-reduce', action='store_true', help='Forward to vLLM --disable-custom-all-reduce')
    parser.add_argument(
        '--hf-overrides',
        type=str,
        default=None,
        help='Forward to vLLM --hf-overrides (JSON string), e.g. {"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}',
    )
    args = parser.parse_args()
    os.environ["SERVE_PORT"] = str(args.port)
    os.environ["DASH_PORT"] = str(args.dash_port)
    os.chdir(pwd)
    cmd_parts = [
        "RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=100",
        "exec",
        "serve",
        "run",
        "--name",
        "VLLMMultiDeployment",
        f"{file}:build_app",
        f"model={args.model}",
        f"tensor-parallel-size={args.tp}",
        f"port={args.port}",
        f"max-num-batched-tokens={HARDCODED_MAX_NUM_BATCHED_TOKENS}",
    ]
    if args.max_model_len is not None:
        cmd_parts.append(f"max-model-len={args.max_model_len}")
    if args.gpu_memory_utilization is not None:
        cmd_parts.append(f"gpu-memory-utilization={args.gpu_memory_utilization}")
    if args.enforce_eager:
        cmd_parts.append("enforce-eager=True")
    if args.disable_custom_all_reduce:
        cmd_parts.append("disable-custom-all-reduce=True")
    if args.hf_overrides is not None:
        cmd_parts.append(f"hf-overrides={args.hf_overrides}")
    cmd = " ".join(shlex.quote(x) for x in cmd_parts)
    import subprocess
    p = subprocess.Popen(cmd, shell=True)
    try:
        p.wait()
    except:
        p.terminate()
        print("interrupted")
        pass
