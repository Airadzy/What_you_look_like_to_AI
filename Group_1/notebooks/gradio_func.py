import gradio as gr
import cv2
from pydub import AudioSegment
from io import BytesIO
import numpy as np

import prediction
from narrator import text_generator, create_audio, fetch_and_play_audio


def run_prediction(image, voice, style_of_text):
    camera = cv2.VideoCapture(0)
    return_value, image2 = camera.read()
    camera.release()
    image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(image3, (218, 178), interpolation=cv2.INTER_AREA)
    prediction_string = prediction.run_prediction(resized_img)
    text = text_generator(prediction_string, style_of_text)
    response = create_audio(text)
    audio_data = BytesIO()
    for chunk in response.iter_content(chunk_size=1024):
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


iface = gr.Interface(
    fn=run_prediction,
    inputs=[
        "image",
        gr.Radio(["Danielle", "Felipe"], label="Choose a voice"),
        gr.Textbox(label="Style of text (e.g Donald Trump)")
    ],
    outputs=gr.Audio(label="Output audio"),
    title="Run Prediction",
    description="Take a photo and run prediction on it."
)
iface.launch(share=True)
