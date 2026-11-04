from typing import NamedTuple

import tachyon_library as tyon

import asyncio
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ComboSampler
from transformers import AutoTokenizer


class llm_config(NamedTuple):
    model_name: str
    model_file: str
    gpu_split_ratio: list[int]
    context_size: int

# model_file = "/local/data/neural_net/exl3/gemma-4-26B-A4B-it-4.10bpw",
gemma4_26_a4b = llm_config(
    model_name = "gemma-4-26B-A4B-it-4.10bpw",
    model_file = "/local/data/neural_net/exl3/gemma-4-26B-A4B-it-3.54bpw",
    gpu_split_ratio = [10,12],
    context_size = 32000
)

def serve_gemma4_exl3(cfg: llm_config, host_ip: str, port: int) -> None:
    config = Config.from_directory(cfg.model_file)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cfg.context_size)
    model.load(use_per_device= cfg.gpu_split_ratio , progressbar=True)
    tokenizer = Tokenizer.from_config(config)
    hf_tokenizer = AutoTokenizer.from_pretrained(cfg.model_file)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": cfg.model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        model_id = body.get("model", cfg.model_name)
        created = int(time.time())
        # Format prompt using the model's tokenizer template
        prompt = hf_tokenizer.apply_chat_template(
                body["messages"], tokenize=False, add_generation_prompt=True
        # prompt = tokenizer.hf_chat_template.apply_chat_template(
            # body["messages"], tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)

        job = Job(
            input_ids=input_ids,
            max_new_tokens=body.get("max_completion_tokens", body.get("max_tokens", 30000)),
            stop_conditions=config.eos_token_id_list,
            sampler=ComboSampler(temperature=body.get("temperature", 0.7))
        )
        generator.enqueue(job)

        def build_usage(r: dict) -> dict:
            gen_tps = r["new_tokens"] / r["time_generate"] if r["time_generate"] > 0 else 0.0
            prefill_tps = r["prompt_tokens"] / r["time_prefill"] if r["time_prefill"] > 0 else 0.0
            tyon.tyon_log(
                f"Prompt: {r['prompt_tokens']} tok (cached: {r['cached_tokens']}) "
                f"in {r['time_prefill']:.2f}s ({prefill_tps:.1f} tok/s) | "
                f"Gen: {r['new_tokens']} tok in {r['time_generate']:.2f}s ({gen_tps:.2f} tok/s)"
            )
            return {
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["new_tokens"],
                "total_tokens": r["prompt_tokens"] + r["new_tokens"],
                "cached_tokens": r["cached_tokens"],
                "time_prefill": round(r["time_prefill"], 4),
                "time_generate": round(r["time_generate"], 4),
                "prompt_tokens_per_sec": round(prefill_tps, 2),
                "tokens_per_sec": round(gen_tps, 2),
            }

        if not body.get("stream", False):
            full_text = ""
            usage = {}
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r["stage"] == "streaming" and "text" in r:
                        full_text += r["text"]
                    if r.get("eos"):
                        usage = build_usage(r)
            return {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
            }

        async def stream():
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r["stage"] == "streaming" and "text" in r:
                        chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {"content": r["text"]}, "finish_reason": None}],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0)

                    if r.get("eos"):
                        usage_chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            "usage": build_usage(r),
                        }
                        yield f"data: {json.dumps(usage_chunk)}\n\n"
                        await asyncio.sleep(0)
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    uvicorn.run(app, host=host_ip, port=port)

serve_gemma4_exl3( gemma4_26_a4b, "100.94.72.0", 8010 )
