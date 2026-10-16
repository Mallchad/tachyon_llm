curl -X 'POST' \
  'http://127.0.0.1:4000/v1/audio/speech' \
  -H 'accept: */*' \
  -H "Authorization: Bearer sk-ce75fe12f0b3d5406ffa113e4a1ebff72b5609efbd8d9872c42c1b9644d33f6e" \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "gemini/gemini-3.8-flash-lite-tts",
  "input": "string",
  "voice": "kore",
  "response_format": "wav",
  "speed": 1,
  "seed": 0
}' \
--output tts_audio.wav
