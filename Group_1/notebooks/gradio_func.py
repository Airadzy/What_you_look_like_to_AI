import gradio as gr
import cv2
import prediction
from narrator import text_generator, create_audio, fetch_and_play_audio


def run_prediction(text):
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    camera.release()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(image, (218, 178), interpolation=cv2.INTER_AREA)
    prediction_string = prediction.run_prediction(resized_img)
    text = text_generator(prediction_string)
    create_audio(text)
    fetch_and_play_audio()


iface = gr.Interface(
    fn=run_prediction,
    inputs="image",
    outputs="text",
    title="Run Prediction",
    description="Take a photo and run prediction on it."
)

iface.launch()
