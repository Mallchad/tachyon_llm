import asyncio
import queue
import time

import numpy as np
import sounddevice as sd
from google import genai


class NanoTimer:
    def __init__(self):
        self.start_time = time.perf_counter_ns()  # Autostart on init
        self.end_time = 0

    def start(self):
        self.start_time = time.perf_counter_ns()
        return self.start_time

    def stop(self):
        self.end_time = time.perf_counter_ns()
        return self.elapsed_ns()

    def elapsed_ns(self):
        return self.end_time - self.start_time

    def elapsed_ms(self):
        return self.elapsed_ns() / 1_000_000

    def stop_and_log(self, label="Operation"):
        duration_ns = self.stop()
        duration_ms = self.elapsed_ms()
        print(f"[Profiler] {label}: {duration_ns:,} ns ({duration_ms:.3f} ms)")
        return duration_ns

# Automatically picks up from GEMINI_API_KEY environment variable
client = genai.Client()

model = "gemini-3.8-live"
config = {"response_modalities": ["AUDIO"]}

async def main():
    audio_queue = queue.Queue()
    gemini_samplerate = 24000
    channels = 1


    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(status, flush=True)
        try:
            # Pull exactly the number of frames requested by the audio driver
            data = audio_queue.get_nowait()
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = 0
            else:
                outdata[:] = data[:len(outdata)]
        except queue.Empty:
            # If network lag happens, output silence instead of crackling garbage
            outdata.fill(0)
            print( "No audio to play...")

    with sd.OutputStream(
            samplerate=gemini_samplerate,
            channels=channels,
            dtype='int16',
            callback = audio_callback
    ):
        async with client.aio.live.connect(model=model, config=config) as session:
            print("Session started")
            # Send content...

            request_timer = NanoTimer()
            request_timer.stop_and_log()
            prompt = "Hello, how are you?"
            print( "sending prompt", prompt )
            await session.send_realtime_input(text=prompt)

            async for response in session.receive():
               if response.server_content and response.server_content.model_turn:
                   for part in response.server_content.model_turn.parts:
                       if part.inline_data:
                           chunk_timer = NanoTimer()
                           audio_data = part.inline_data.data
                           chunk_timer.stop_and_log( "Receiving Audio" )
                           # Process or play the audio data
                           print(f"Received audio chunk: {len(audio_data)} bytes")
                           # Convert raw bytes to a numpy array of 16-bit integers
                           audio_np = np.frombuffer(audio_data, dtype=np.int16)
                           chunk_timer.stop_and_log( "Processed audio" )

                           # Write chunks directly to your speakers in real-time
                           audio_queue.put(audio_np)
                           print( audio_queue.len )
                           chunk_timer.stop_and_log( "played audio" )

                   if response.server_content and response.server_content.turn_complete:
                       break

            request_timer.stop_and_log( "Request Complete")

if __name__ == "__main__":
    asyncio.run(main())
