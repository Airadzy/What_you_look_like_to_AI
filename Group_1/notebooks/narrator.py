import os
from openai import OpenAI
import requests
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import json
from io import BytesIO
import tempfile
import time

import take_img
import simpleaudio

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Set environment variables for FFmpeg
os.environ["PATH"] += os.pathsep + config["ffmpeg_path"]
os.environ["FFMPEG"] = os.path.abspath(config["ffmpeg_exe"])
os.environ["FFPROBE"] = os.path.abspath(config["ffprobe_exe"])


def text_generator(prediction_string, style):
    # Use the API key from config
    style = 'in the style of' + style
    client = OpenAI(api_key=config["openai_api_key"])
    # Setting max_tokens to 256 to approximate a limit of 1024 characters
    max_tokens = 100
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": f"create a text that describes a person with the following attributes {style}"},
            {"role": "user", "content": f"{prediction_string}"}
        ],
        max_tokens=max_tokens  # Add the max_tokens parameter here
    )
    print(completion.choices[0])
    return completion.choices[0].message.content


def create_audio(text):
    CHUNK_SIZE = 1024
    payload = {
        "text": f"{text}",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    headers = {
        "xi-api-key": config["elevenlabs_api_key"],
        "Content-Type": "application/json"
    }


    response = requests.post(config["elevenlabs_url"], json=payload, headers=headers)

    # Create an in-memory buffer to store audio data
    audio_data = BytesIO()
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            audio_data.write(chunk)

    # Seek to the beginning of the buffer
    audio_data.seek(0)

    # Load the audio data as an AudioSegment
    audio = AudioSegment.from_file(audio_data)

    # Convert AudioSegment to NumPy array
    audio_np = audio.get_array_of_samples()

    return response

def fetch_and_play_audio(text):
    audio_file_path = create_audio(text)
    audio = AudioSegment.from_mp3(audio_file_path)
    _play_with_simpleaudio(audio)

# def main():
#     # take photo
#     image = take_img.take_photo()
#
#     # Run prediction
#     prediction_string = prediction.run_prediction(image)
#
#     # generate text and text2audio
#     print("👀 David is watching...")
#     generated_text = text_generator(prediction_string)
#     create_audio(generated_text)
#     fetch_and_play_audio()
#
#
# if __name__ == "__main__":
#     main()
