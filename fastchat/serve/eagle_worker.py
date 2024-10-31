"""
A model worker that executes the model based on Eagle.

"""

import argparse
import asyncio
import json
from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import torch
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length

import uuid
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from .eagle_model.ea_model import EaModel

app = FastAPI()

class EagleWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        ea_model_path: str,
        eval_type: bool,
        decoding: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        conv_template: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )
        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: Eagle worker..."
        )
        self.init_heart_beat()
        

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.decoding = decoding
        self.eval_type = eval_type
        logger.info(
            f"eval_type: {self.eval_type}, decoding: {self.decoding}"
        )
        if self.eval_type == 'baseline':
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", trust_remote_code=True
            ).eval()
        elif self.eval_type == 'eagle':
            self.model = EaModel.from_pretrained(
                base_model_path=model_path,
                ea_model_path=ea_model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto"
            ).eval()
        else:
            raise ValueError("eval_type should be 'baseline' or 'eagle'")

        self.context_len = self.model.config.max_position_embeddings
        if self.decoding == 'greedy':
            self.model.generation_config.do_sample = False  # use greedy decoding
        elif self.decoding == 'sampled':
            self.model.generation_config.do_sample = True  # use stochastic decoding
        else:
            raise ValueError("decoding should be 'greedy' or 'sampled'")



    def count_token(self, params):
        prompt = params["prompt"]

        input_ids = self.tokenizer.tokenize(prompt)
        input_echo_len = len(input_ids)

        ret = {
            "count": input_echo_len,
            "error_code": 0,
        }
        return ret
    
    async def generate(self, params):
        SEED = 102 # 可以是任意整数，这里以42为例

        def set_random_seeds(seed):
            torch.manual_seed(seed)          # 设置CPU上的PyTorch随机数生成器种子
            torch.cuda.manual_seed(seed)     # 设置当前GPU上的随机数生成器种子
            torch.cuda.manual_seed_all(seed)  # 如果有多个GPU，设置所有GPU上的随机数生成器种子
            np.random.seed(seed)              # 设置NumPy的随机数生成器种子
            random.seed(seed)                 # 设置Python内置random模块的随机数生成器种子

        # 调用函数设置种子
        set_random_seeds(SEED)

        input_ids = self.tokenizer.encode(params['prompt'], return_tensors="pt").cuda()
        if self.eval_type == 'baseline':
            response = self.model.generate(input_ids, return_dict_in_generate=True, output_scores=True)
        elif self.eval_type == 'eagle':
            if self.decoding == 'greedy':
                response = self.model.eagenerate(self.tokenizer, input_ids, temperature=0.0, max_length=8048, max_new_tokens=2048, verbose=False)
            elif self.decoding == 'sampled':
                response = self.model.eagenerate(self.tokenizer, input_ids, temperature=1.0, max_length=8048, max_new_tokens=2048, top_p=0.5, verbose=False)
            else:
                raise ValueError("decoding should be 'greedy' or 'sampled'")
        else:
            raise ValueError("eval_type should be 'baseline' or 'eagle'")

        def _decode_chatml(
            tokens,
            *,
            stop_words,
            eod_token_ids,
            tokenizer,
            raw_text_len: int,
            context_length: int,
            verbose: bool = False,
            return_end_reason: bool = False,
            errors: str='replace'
        ):
            end_reason = f"Gen length {len(tokens)}"
            eod_token_idx = context_length
            for eod_token_idx in range(context_length, len(tokens)):
                if tokens[eod_token_idx] in eod_token_ids:
                    end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
                    break

            trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
            if verbose:
                # print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
                # print("\nRaw Generate:", trim_decode_tokens)
                print("\nEnd Reason:", end_reason)
            for stop_word in stop_words:
                trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
            trim_decode_tokens = trim_decode_tokens.strip()
            if verbose:
                print("\nGenerate:", trim_decode_tokens)

            if return_end_reason:
                return trim_decode_tokens, end_reason
            else:
                return trim_decode_tokens

        def decode_tokens(
            tokens,
            tokenizer,
            raw_text_len: int,
            context_length: int,
            verbose: bool = False,
            return_end_reason: bool = False,
            errors: str="replace",
        ) -> str:
            if torch.is_tensor(tokens):
                tokens = tokens.cpu().numpy().tolist()
            return _decode_chatml(
                tokens,
                stop_words=[],
                eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
                tokenizer=tokenizer,
                raw_text_len=raw_text_len,
                context_length=context_length,
                verbose=verbose,
                return_end_reason=return_end_reason,
                errors=errors,
            )

        output_tokens = response[0] if torch.is_tensor(response) else response[0][0]
        output = decode_tokens(
            output_tokens,
            self.tokenizer,
            raw_text_len=len(params['prompt']),
            context_length=len(input_ids[0]),
            verbose=False,
            errors='replace'
        )

        # logger.info(f"output: {output}")
        return {
            "text": output,
            "error_code": 0,
        }           

def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    worker: EagleWorker = app.state.worker
    await acquire_worker_semaphore()
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)

@app.post("/worker_get_status")
async def api_get_status(request: Request):
    worker: EagleWorker = app.state.worker
    return worker.get_status()

@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    worker: EagleWorker = app.state.worker
    return worker.count_token(params)

@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    worker: EagleWorker = app.state.worker
    return worker.get_conv_template()

@app.post("/model_details")
async def api_model_details(request: Request):
    worker: EagleWorker = app.state.worker
    return {"context_length": worker.context_len}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.3")
    parser.add_argument("--ea-model-path", type=str, default="/cpfs01/shared/public/yyang/megatron_lm_workspace/hf-checkpoint/eagle/qwen_sft_test_0301_v1/state_20")
    parser.add_argument("--eval_type", type=str, default="baseline")
    parser.add_argument("--decoding", type=str, default="greedy")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    args = parser.parse_args()
    worker = EagleWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.ea_model_path,
        args.eval_type,
        args.decoding,
        args.model_names,
        args.limit_worker_concurrency,
        args.conv_template,
    )

    app.state.worker = worker
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")