
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
# Stolen by jitsi meet
# port: int = 8080
port: int = 8006
# host_ip = "192.168.1.172"
host_ip = "100.94.72.0"

gemma4_e4b_f16 = llm_config(
    "gemma-4-E4B-BF16",
    "/local/data/neural_net/gguf/gemma-4-E4B-it-BF16.gguf",
    # gpu_split_rtx3060_only,
    # gpu_split_rtx3080_only,
    "8,12",
    64000
)

gemma4_e4b_q8 = llm_config(
    "gemma-4-E4B-Q8_0",
    "/local/data/neural_net/gguf/gemma-4-E4B-it-Q8_0.gguf",
    # gpu_split_rtx3060_only,
    gpu_split_rtx3080_only,
    64000
)

gemma4_e2b_q4 = llm_config(
    "gemma-4-E2B-it-Q4_0",
    "/local/data/neural_net/gguf/gemma-4-E2B-it-Q4_0.gguf",
    # "10,12",
    gpu_split_rtx3080_only,
    # gpu_split_rtx3060_only,
    64000
)
gemma4_26_a4b = llm_config(
    "gemma-4-26B-A4B-it-Q4_0",
    "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf",
    # "6000,12000",
    "8000,12000",
    context_size = 128000
)

gemma4_26_a4b_q5 = llm_config(
    "gemma-4-26B-A4B-it-UD-Q5_K_S",
    "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf",
    gpu_split_ratio = "80,120",
    context_size = 32000
)


qwen3_a3b_q4 = llm_config(
    "Qwen3-Coder-30B-A3B-Instruct-1M-IQ4_XS",
    "/local/data/neural_net/gguf/Qwen3-Coder-30B-A3B-Instruct-1M-IQ4_XS.gguf",
    gpu_split_ratio = "7000,12000",
    # "8000,12000",
    context_size = 32000
)
# model = gemma4_e2b_f16

def run_gemma4_moe() -> None:
    model = gemma4_26_a4b

    command_args = [
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--flash-attn", "on",

        "--threads", "6",
        "--no-mmap",
        "--mlock",
        # No parallel processing (parallel user slots)
        "--parallel", "1",
        "--threads-batch", "6",
        "--batch-size", "2048",
        "--ubatch-size", "512",

        "--main-gpu", "0",
        # "--verbose",
        # "--gpu-layers", "all",
        # "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        "--reasoning", "off",
        # Pointless, only relevant for llama.ui
        # "--tools", "read_file,grep_search",

        # Multi-Modal Projector
        "--mmproj", "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-mmproj-BF16.gguf",
        # Doesn't fit unless we push the mmproj to the CPU, it slows down image processing a lot.
        "--no-mmproj-offload",

        # Network Settings
        "--host", host_ip,
        "--port", str(port)
    ]

    speculative_decoding_args = [
        # Enable multi-token speculative prediction for speedup
        "--spec-draft-model", "/local/data/neural_net/gguf/mtp-gemma-4-26B-A4B-it-Q8_0.gguf",
        # "--spec-draft-model", "/local/data/neural_net/gguf/mtp-gemma-4-26B-A4B-it-BF16.gguf",
        "--spec-type", "draft-mtp",
        # Number of tokens speculatively predicted, variable speedup
        "--spec-draft-n-max", "1",
        "--spec-draft-type-k", "q5_0",
        "--spec-draft-type-v", "q5_0",
    ]
    speculative_decoding_on: bool = False
    if speculative_decoding_on:
        command_args += speculative_decoding_args
    tyon.execute_list( command_args )

def run_gemma4_moe_q5_k_s() -> None:
    model = gemma4_26_a4b_q5
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--flash-attn", "on",

        # Enable multi-token prediction for speed
        # TODO: Why is this not allowign draft-mtp?
        # "--spec-type", "draft-simple",
        # "--spec-type", "draft-mtp",
        # "--spec-draft-n-max", "2",

        # No parallel processing
        "--parallel", "1",
        "--main-gpu", "0",
        "--verbose",
        "--gpu-layers", "30",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning-format", "deepseek",
        "--reasoning", "off",
        "--tensor-split", model.gpu_split_ratio,

        # Multi-Modal Projector
        # "--mmproj", "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-mmproj-BF16.gguf",
        # Doesn't fit unless we push the mmproj to the CPU, it slows down image processing a lot.
        # "--no-mmproj-offload",

        # Network Settings
        "--host", host_ip,
        "--port", str(port)
    ] )


def run_gemma4_e4b_q8() -> None:
    model = gemma4_e4b_q8
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,

        "--parallel", "1",
        # Doesn't fit in memory when I use 96000
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--flash-attn", "on",
        "--main-gpu", "0",

        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--spec-draft-type-k", "bf16",
        "--spec-draft-type-v", "bf16",

        # Enable multi-token prediction for speed
        # "--spec-draft-model", "/local/data/neural_net/gguf/mtp-gemma-4-E4B-it-Q8_0.gguf",
        "--spec-draft-model", "/local/data/neural_net/gguf/mtp-gemma-4-E4B-it-BF16.gguf",
        "--spec-type", "draft-mtp",
        # Number of tokens predicted, variable speedup
        "--spec-draft-n-max", "3",

        # does verbose slow generation?
        # "--verbose",

        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning-format", "deepseek",
        "--reasoning", "off",
        "--tensor-split", model.gpu_split_ratio,
        "--host", host_ip,
        "--port", str(port)
    ] )

def run_gemma4_e2b_q4() -> None:
    model = gemma4_e2b_q4
    raise Exception()

def run_qwen3_a3b() -> None:
    model = qwen3_a3b_q4
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--cache-ram", "32000",

        "--flash-attn", "on",

        # No parallel processing
        "--parallel", "1",
        "--main-gpu", "0",
        "--verbose",
        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning", "off",

        "--split-mode", "layer", # defaut
        "--tensor-split", model.gpu_split_ratio,

        # Multi-Modal Projector
        # "--mmproj", "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-mmproj-BF16.gguf",
        # Doesn't fit unless we push the mmproj to the CPU, it slows down image processing a lot.
        "--no-mmproj-offload",

        # Network Settings
        "--host", host_ip,
        "--port", str(port)
    ] )

qwen3_8_27B_q4 = llm_config(
    "Qwen3.8-27B-UD-IQ4_XS",
    "/local/data/neural_net/gguf/Qwen3.8-27B-UD-IQ4_XS.gguf",
    gpu_split_ratio = "7000,12000",
    # "8000,12000",
    context_size = 16000
)
def run_qwen3_8_27b() -> None:
    model = qwen3_8_27B_q4
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "bf16",
        "--cache-type-v", "bf16",
        "--cache-ram", "32000",

        "--flash-attn", "on",

        # "--batch-size", "2048",
        # "--ubatch-size", "512",

        # Enable multi-token prediction for speed
        # It is indeed faster with dense models
        "--spec-draft-model", "/local/data/neural_net/gguf/mtp-Qwen3.8-27B-Q4_0.gguf",
        "--spec-type", "draft-mtp",
        # Number of tokens predicted, variable speedup
        "--spec-draft-n-max", "1",
        # "--spec-draft-n-max", "2",

        # No parallel processing
        "--parallel", "1",
        "--main-gpu", "0",
        # "--verbose",
        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        "--reasoning", "on",

        "--split-mode", "layer", # defaut
        "--tensor-split", model.gpu_split_ratio,

        # Multi-Modal Projector
        # "--mmproj", "/local/data/neural_net/gguf/gemma-4-26B-A4B-it-mmproj-BF16.gguf",
        # Doesn't fit unless we push the mmproj to the CPU, it slows down image processing a lot.
        "--no-mmproj-offload",

        # Network Settings
        "--host", host_ip,
        "--port", str(port)
    ] )

talkie_1930_q8 = llm_config(
    "talkie-1930-13b-it-Q8_0",
    "/local/data/neural_net/gguf/talkie-1930-13b-it-Q8_0.gguf",
    # gpu_split_rtx3060_only,
    gpu_split_full_both,
    16000
)

def run_talkie_1930_q8() -> None:
    model = talkie_1930_q8
    tyon.execute_list([
        "/usr/bin/llama-server",
        "--alias", model.model_name,
        "--model", model.model_file,
        # "--verbose",

        "--parallel", "1",
        # Doesn't fit in memory when I use 96000
        "--models-max", "1",
        "-c", str(model.context_size),
        # NOTE: Accuracy suffers really badly if you quantize the kv cache
        "--cache-type-k", "q8_0",
        "--cache-type-v", "bf16",
        "--flash-attn", "on",
        "--main-gpu", "0",

        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--spec-draft-type-k", "bf16",
        "--spec-draft-type-v", "bf16",
        "--gpu-layers", "all",
        "--log-prompts-dir", "~/tmp/llama_server_prompts.log",
        # "--reasoning-format", "deepseek",
        "--tensor-split", model.gpu_split_ratio,
        "--host", host_ip,
        "--port", str(port)
    ] )



run_gemma4_moe()
# run_qwen3_a3b()
# run_qwen3_8_27b()
# run_gemma4_moe_q5_k_s()
# run_gemma4_e4b_q8()

# Needs talkie PR
# run_talkie_1930_q8()
