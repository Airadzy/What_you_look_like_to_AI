import gradio as gr
import cv2
from Group_1.application import prediction
from Group_1.application.narrator import text_generator, create_audio


def run_prediction(image2, voice, style_of_text):
    """
    Gets the inputs from gradio in order to run the prediction
    Returns the voice over the predictions describing the person
    @param image2:
    @type image2:
    @param voice:
    @type voice:
    @param style_of_text: input from gradio to sytle the text prediction
    @type style_of_text: string
    @return: Audio of voice over predictions
    @rtype: audio mp3
    """
    camera = cv2.VideoCapture(0)
    return_value, image = camera.read()
    camera.release()
    resized_img = cv2.resize(image, (218, 178), interpolation=cv2.INTER_AREA)
    #cv2.imwrite('saved_image.jpg', resized_img)
    prediction_string = prediction.run_prediction(resized_img)
    text = text_generator(prediction_string, style_of_text)
    audio_bytes = create_audio(text)
    return audio_bytes


def main():
    iface = gr.Interface(
        fn=run_prediction,
        inputs=[
            gr.Image(label="Input Image"),
            gr.Radio(["Danielle", "Felipe"], label="Choose a voice"),
            gr.Textbox(label="Style of text (e.g Donald Trump)")
        ],
        outputs=gr.Audio(label="Output audio"),
        title="Run Prediction",
        description="Take a photo and run prediction on it."
    )
    iface.launch(share=True)


if __name__ == "__main__":
    main()
