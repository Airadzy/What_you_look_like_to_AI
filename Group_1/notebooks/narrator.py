import os
from openai import OpenAI
import requests
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import json
import time
import prediction
import take_img
import simpleaudio

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Set environment variables for FFmpeg
os.environ["PATH"] += os.pathsep + config["ffmpeg_path"]
os.environ["FFMPEG"] = os.path.abspath(config["ffmpeg_exe"])
os.environ["FFPROBE"] = os.path.abspath(config["ffprobe_exe"])

def text_generator(prediction_string):
    # Use the API key from config
    client = OpenAI(api_key=config["openai_api_key"])

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "create a text that describes a person with the following attributes in the style of Donald Trump"},
            {"role": "user", "content": f"{prediction}"}
        ]
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
    with open('output.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def fetch_and_play_audio():
    audio = AudioSegment.from_mp3(config["audio_file_path"])
    _play_with_simpleaudio(audio)

def main():

    # take photo
    image = take_img.take_photo()

    # Run prediction
    prediction_string = prediction.run_prediction(image)

    # generate text and text2audio
    print("👀 David is watching...")
    generated_text = text_generator(prediction_string)
    create_audio(generated_text)
    fetch_and_play_audio()

if __name__ == "__main__":
    main()
