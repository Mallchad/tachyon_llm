import requests
import json
import os

# --- Configuration ---
API_URL = 'http://127.0.0.1:8004/v1/audio/speech'
OUTPUT_FILE = '/home/mallchad/tmp/speech.wav'  # Note: Ensure the 'tmp' directory exists or adjust the path

# --- Input Data ---
# IMPORTANT: You need to replace "string" with the actual string you want to convert.
input_text = "This is the text I want to convert to speech."

# --- Request Payload ---
payload = {
    "model": "Emily.wav",
    "input": input_text,  # Replace "string" with your actual text
    "voice": "Emily.wav",
    "response_format": "wav",
    "speed": 1,
    "seed": 0
}

print(f"Sending request to: {API_URL}")
print(f"Input text: '{input_text[:50]}...'")

try:
    # Make the POST request
    response = requests.post(
        API_URL,
        headers={'accept': '*/*', 'Content-Type': 'application/json'},
        data=json.dumps(payload)
    )

    # Check if the request was successful (status code 200-299)
    response.raise_for_status()

    # Save the binary content to a file
    with open(OUTPUT_FILE, 'wb') as f:
        f.write(response.content)

    print(f"\n✅ Success! Audio saved to: {os.path.abspath(OUTPUT_FILE)}")

except requests.exceptions.HTTPError as errh:
    print(f"\n❌ HTTP Error occurred: {errh}")
    print(f"Response content: {response.text}")
except requests.exceptions.ConnectionError as errc:
    print(f"\n❌ Error Connecting: {errc}")
    print("Please ensure the server at http://127.0.0.1:8004 is running.")
except requests.exceptions.RequestException as err:
    print(f"\n❌ An unexpected error occurred: {err}")
