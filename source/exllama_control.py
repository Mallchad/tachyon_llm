from typing import NamedTuple

import tachyon_library as tyon

import json
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
    gpu_split_ratio = [11,13],
    context_size = 8192
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

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        # Format prompt using the model's tokenizer template
        prompt = hf_tokenizer.apply_chat_template(
                body["messages"], tokenize=False, add_generation_prompt=True
        # prompt = tokenizer.hf_chat_template.apply_chat_template(
            # body["messages"], tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(prompt, add_bos=False)

        job = Job(
            input_ids=input_ids,
            max_new_tokens=body.get("max_completion_tokens", body.get("max_tokens", 2048)),
            stop_conditions=[tokenizer.eos_token_id],
            sampler=ComboSampler(temperature=body.get("temperature", 0.7))
        )
        generator.enqueue(job)

        async def stream():
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r["stage"] == "streaming" and "text" in r:
                        chunk = {"choices": [{"delta": {"content": r["text"]}}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    uvicorn.run(app, host=host_ip, port=port)

serve_gemma4_exl3( gemma4_26_a4b, "127.1", 8010 )
