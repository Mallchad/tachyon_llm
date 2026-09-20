# from torch,quantization
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print( "Starting Experiments" )

model_id = "/local/data/neural_net/gemma-4-E4B-it"

memory_config = { 0: "6GiB", 1: "8GiB" }

# 1. Load model in BF16 (Ampere's native high-precision format)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="auto",
    max_memory = memory_config


)
# Try to quantize the model
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype = torch.quint8 )

# model_quantized = torch.quantization.quantize_dynamic(
#     model_id, {torch.nn.Linear}, dtype = torch.bfloat16 )

# model.max_position_embeddings= 4096

tokenizer = AutoTokenizer.from_pretrained( model_id )

user_prompt: str = "Write a haiku about the tortoise and the hare"
# prompt: str = "a b c d"
prompt_dict = [{"role": "user", "content": user_prompt }]
prompt = tokenizer.apply_chat_template(
    prompt_dict, tokenize=False, add_generation_prompt=True
)
# prompt = "<|user|> Write a hikou about the tortoise and the hare \n<|end|> <|assistant|>"
inputs = tokenizer( prompt, return_tensors = "pt" ).to("cuda")

print( "Token Vector: ", inputs )


for i in range(1):
    with torch.no_grad():
        outputs = model_quantized.generate(
        **inputs,
        max_new_tokens = 10,
        use_cache = False,
        do_sample = True,
        temperature = 1.0
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print( "outputs", outputs )
        print( f"Prompt Used: {inputs} \nResponse: {response}" )


# print( "Waiting before exit" )
time.sleep( 10 )


print( "Exiting program" )
