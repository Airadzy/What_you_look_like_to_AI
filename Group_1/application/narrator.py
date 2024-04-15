import os
from openai import OpenAI
import requests
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio
import json
from io import BytesIO
import numpy as np

with open('config.json') as config_file:
    config = json.load(config_file)

def text_generator(prediction_string, style,celebrity_name):
    """
    Gets a string with a text with the features and 'yes' or 'no' for each feature
    Gets a string 'style' which is an input from gradio so people can customize the text
    in the style of a famous person for example.
    Turns the inputs into text using OPEN AI API and returns it
    @param prediction_string: predicted features from model
    @type prediction_string: string
    @param style: input from gradio
    @type style: string
    @return: OPEN AI generated text
    @rtype: string
    """
    style = 'in the style of' + style
    client = OpenAI(api_key=config["openai_api_key"])

    max_tokens = 100
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": f"create a text that describes a person with the following attributes in the style of {style}. It should start like: The person we chose for you is close to {celebrity_name} with (and then describe the attributes)"},
            {"role": "user", "content": f"{prediction_string}"}
        ],
        max_tokens=max_tokens
    )
    print(completion.choices[0])
    return completion.choices[0].message.content


def create_audio(text):
    """
    Gets the text from text_generator() and creates the audio using ELEVENLABS API
    Returns the audio format in mp3 to listen on the gradio App
    @param text: generated text from OPEN AI
    @type text: string
    @return: Voice over the text generated on OPEN AI
    @rtype: audio
    """
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
    audio_np = np.array(audio.get_array_of_samples())

    # Convert the NumPy array back to an AudioSegment
    audio_output = AudioSegment(audio_np.tobytes(), frame_rate=audio.frame_rate, sample_width=audio.sample_width,
                                channels=audio.channels)

    # Convert the AudioSegment to bytes
    audio_bytes = audio_output.export(format="mp3").read()
    return audio_bytes


def fetch_and_play_audio(text):
    """
    @param text:
    @type text:
    @return:
    @rtype:
    """
    audio_file_path = create_audio(text)
    audio = AudioSegment.from_mp3(audio_file_path)
    _play_with_simpleaudio(audio)

