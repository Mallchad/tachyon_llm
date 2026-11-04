import json
from typing import NamedTuple
import requests

import tachyon_library as tyon

exllama_base_url = "http://100.94.72.0:8010/"
base_url = exllama_base_url
endpoint_completions = "v1/chat/completions"
endpoint_models = "v1/models"

endpoint_url = endpoint_completions
request_url = exllama_base_url + endpoint_url
tyon.tyon_log( "Endpoint:", request_url )

# curl "https://api.groq.com/openai" \
#   -X POST \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer ${GROQ_API_KEY}" \
#   -d '{
#          "messages": [
#            {
#              "role": "user",
#              "content": "Write a poem about cherry blossoms"
#            }
#          ],
#          "model": "qwen/qwen3.8-27b",
#          "temperature": 1,
#          "max_completion_tokens": 2048,
#          "top_p": 1,
#          "stream": false,
#          "reasoning_effort": "medium",
#          "stop": null
#        }'

# curl "https://api.groq.com/openai/v1/models"

api_key = "xxx"

prompt_unsanitized = "Write a poem about cherry blossoms"
prompt_unsanitized = ("""Write a very simple Vulkan program that draws a cube with X11 in C++.
keep it in a single file.
Do not talk to the user they will not be reading the content and will not response after the fact.
Do do not behave like an assistant, simply do the task.

Use snake_case formatting
Use expanded GNU style function brackets add( 5 ) instead of add(5)
Use initializer lists instead of member access operators where possible to take advantage of explicit intialization rules
Typedef all Vulkan types to a snake_case equivilant
Use Xlib not xcb, nobody uses xcb because nobody understands it as Xlib is well documented and xcb is not.
"""
)
# Sanitize newlines chars for json
prompt = prompt_unsanitized.encode("unicode_escape").decode("utf-8")
print( "Prompt:", prompt_unsanitized )

json_payload = '''{
         "messages": [
           {
             "role": "user",
             "content": "%s"
           }
         ],
         "model": "qwen/qwen3.8-27b",
         "temperature": 0.8,
         "max_completion_tokens": 30000,
         "top_p": 1,
         "stream": true,
         "reasoning_effort": "low"
       }''' % prompt
tyon.tyon_log( "json_payload: ", json_payload )

def request_curl() -> None:
    tyon.execute_list( [
        "curl",
        request_url,
        "--request", "POST",
        "--header", "Content-Type: application/json",
        "--header", f"Authorization: Bearer {api_key}",
        "--data", json_payload
    ])

def request_python() -> None:
    log_path = "../transient/logs/openai_stream.log"
    usage = None
    with requests.post(
        request_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        data=json_payload,
        stream=True,
    ) as response, open(log_path, "w", encoding="utf-8") as log_file:
        for line in response.iter_lines(chunk_size=None, decode_unicode=True):
            if not line:
                continue
            log_file.write(line + "\n")
            log_file.flush()

            data_str = line.removeprefix("data: ").strip()
            if data_str == "[DONE]":
                break

            chunk = json.loads(data_str)
            if "usage" in chunk:
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            content = choices[0].get("delta", {}).get("content", "") if choices else ""
            if content:
                # flush = true is not necessary but it makes it much lower
                # latency at the cost of CPU usage
                print(content, end="", flush=True)
    print()
    if usage:
        tyon.tyon_log(
            f"Generation complete: {usage.get('completion_tokens')} tokens in "
            f"{usage.get('time_generate')}s ({usage.get('tokens_per_sec')} tok/s) | "
            f"Prompt: {usage.get('prompt_tokens')} tokens ({usage.get('prompt_tokens_per_sec')} tok/s)"
        )

# request_curl()
request_python()
