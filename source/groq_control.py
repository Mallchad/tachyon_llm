

curl "https://api.groq.com/openai/v1/chat/completions" \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GROQ_API_KEY}" \
  -d '{
         "messages": [
           {
             "role": "user",
             "content": "Write a poem about cherry blossoms"
           }
         ],
         "model": "qwen/qwen3.8-27b",
         "temperature": 1,
         "max_completion_tokens": 2048,
         "top_p": 1,
         "stream": false,
         "reasoning_effort": "medium",
         "stop": null
       }'


curl "https://api.groq.com/openai/v1/models"
