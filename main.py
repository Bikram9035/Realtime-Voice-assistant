import os
import pyaudio
import wave
import keyboard
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
import time

# Load environment variables from .env file
load_dotenv(".env")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# Initialize clients
groqllm = Groq()
openai_client = OpenAI()

# Initialize PyAudio
p = pyaudio.PyAudio()

def record_audio():
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=10000, input=True, frames_per_buffer=256)
    audio_buffer = bytearray()

    print("Press space to start recording. Press space again to stop.")
    keyboard.wait('space')
    print("Recording...")

    start_time = time.time()
    while True:
        audio_chunk = stream.read(256)
        audio_buffer.extend(audio_chunk)
        if keyboard.is_pressed('space') and time.time() - start_time > 0.5:
            break

    print("Recording stopped.")
    stream.stop_stream()
    stream.close()

    return audio_buffer
def transcribe_audio(audio_chunks):
    # Open a new audio stream to write the chunks and send them to the API
    with wave.open("recorded_audio.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        
        # for chunk in audio_chunks:
        wf.writeframes(audio_chunks)

    transcription = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=open("recorded_audio.wav", "rb"),
        language="en"
        
    )
    return transcription.text

def generate_response(text):
    response = groqllm.chat.completions.create(
        model="llama3-8b-8192",
        # model="mixtral-8x7b-32768",
        # model="llama3-70b-8192",
        # model="gemma-7b-it",
        temperature=0.7,
        max_tokens=350,
        stream=True,
        messages=[
            {"role": "system", "content": """ you are a teenager who answers in super short and crisp manner.
                                              you use uhh, hmm, ohh as pause word. 
                                              you are naturally very expressive """},  
            {"role": "user", "content": text}         
        ]
    )

    response_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            response_text += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content

def text_to_speech_stream(text):
    player = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

    with openai_client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format='pcm',
    ) as response:
        silence_threshold = 0.01
        stream_start = False

        for chunk in response.iter_bytes(chunk_size=256):
            if stream_start:
                player.write(chunk)
            elif max(chunk) > silence_threshold:
                stream_start = True
                player.write(chunk)

    player.close()

def main():
    while True:
        
        audio_chunks = record_audio()
        # start=time.time()
        transcription = transcribe_audio(audio_chunks)
        print(f"User: {transcription}")

        response_generator = generate_response(transcription)
        response_text = "".join(response_generator)
        print(f"David: {response_text}")
        # end=time.time()
        # print(f"latency:{end-start}")
        text_to_speech_stream(response_text)
        

if __name__ == "__main__":
    main()
