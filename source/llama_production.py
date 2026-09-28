
from typing import NamedTuple

import tachyon_library as tyon

# execute_list([
#     /usr/bin/llama-server
#     "--model", "/local/data/neural_net/gguf/gemma-4-E2B-it-BF16.gguf",
#     "--models-max", "1 -c 128000",
#     "--cache-type-k", "q4_0",
#     "--cache-type-v", "f16",
#     "--flash-attn", "off",
#     "--main-gpu", "0",
#     "--verbose",
#     "--gpu-layers", "all",
#     "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
#     "--reasoning-format", "deepseek",
#     "--reasoning", "off",
#     "--tensor-split", "12,0",
#     "--tools", "read_file,grep_search",
#     "--port", "8080" ]
#              )

class llm_config(NamedTuple):
    model_name: str
    model_file: str
    gpu_split_ratio: str
    context_size: int

gpu_split_rtx3080_only = "1,0"
gpu_split_rtx3060_only = "0,12"
# Has to be a bit less than 8 due to weird skew towards the 3080
gpu_split_full_both = "8, 12"
gpu_split_rtx3080_dominant = "10,4"
gpu_split_rtx3060_dominant = "6,12"
gpu_split_ratio: str = gpu_split_full_both
# port: int = 8002
port: int = 8080
host_ip = "192.168.1.172"

gemma4_e4b_f16 = llm_config(
    "gemma-4-E4B-BF16",
    "/local/data/neural_net/gguf/gemma-4-E2B-it-BF16.gguf",
    gpu_split_rtx3060_only,
    64000
)

gemma4_e2b_q4 = llm_config(
    "gemma-4-E2B-it-Q4_0",
    "/local/data/neural_net/gguf/gemma-4-E2B-it-Q4_0.gguf",
    gpu_split_rtx3060_only,
    64000
)
gemma4_26_a4b = llm_config(
    "gemma-4-26B-A4B-it-Q4_0",
    "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-Q4_0.gguf",
    gpu_split_full_both,
    128000
)


# model = gemma4_e2b_f16

def run_gemma4_moe() -> None:
    model = gemma4_26_a4b
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        "--models-max", "1",
        "-c", "96000",
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--flash-attn", "on",
        # No parallel processing
        "--parallel", "1",
        "--main-gpu", "0",
        "--verbose",
        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning-format", "deepseek",
        "--reasoning", "off",
        "--tensor-split", model.gpu_split_ratio,
        "--tools", "read_file,grep_search",
        "--host", host_ip,
        "--port", str(port)
    ] )

def run_gemma4_e4b_f16() -> None:
    model = gemma4_e4b_f6
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        # Doesn't fit in memory when I use 96000
        "--models-max", "1 -c 96000",
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "q16_0",
        "--cache-type-v", "q16_0",
        "--flash-attn", "on",
        "--main-gpu", "0",
        "--verbose",
        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning-format", "deepseek",
        "--reasoning", "off",
        "--tensor-split", model.gpu_split_ratio,
        "--tools", "read_file,grep_search",
        "--host", host_ip,
        "--port", str(port)
    ] )

def run_gemma4_e2b_q4() -> None:
    model = gemma4_e2b_q4
    raise Exception()


run_gemma4_moe()
# run_gemma4_e2b()
