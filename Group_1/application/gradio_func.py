import gradio as gr
import cv2
from Group_1.application import prediction
from Group_1.application.narrator import text_generator, create_audio
import return_images
import os
import json

# Load configuration from a JSON file
with open('config.json') as config_file:
    config = json.load(config_file)


def run_prediction(voice, celebrity_names):
    """
    Processes the celebrity name input to generate an audio description and return images.

    Args:
       voice (str): The selected voice for generating the audio output.
       celebrity_names (str): The name of the celebrity provided by the user to find and process.

    Returns:
       tuple: Contains the generated audio bytes, the input celebrity image, and the best match image.
              All returned as respective outputs for Gradio interface.
    """
    # Download and verify the image
    image_path = return_images.download_first_image(celebrity_names,config["google_API_key"], config["google_cx"])
    if not image_path or not os.path.exists(image_path):
        return None, "Image not available", "Image not available"

    img = cv2.imread(image_path)
    if img is None:
        return None, "Failed to load image", "Image not available"

    # Process the image for prediction
    resized_img = cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA)
    cv2.imwrite("current_image.jpg", resized_img)
    prediction_string, connections = prediction.run_prediction(resized_img)

    # Find and load the best matching image
    best_match = return_images.find_best_match(connections, config["csv_path"])
    best_match_path = os.path.join(config["images_path"], best_match)
    best_match_img = cv2.imread(best_match_path)
    if best_match_img is None:
        return None, resized_img, "Best match image not available"

    # Convert images for display and generate audio
    best_match_img = cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB)
    text = text_generator(prediction_string, "matchmaker", celebrity_names)
    audio_bytes = create_audio(text)
    input_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    return audio_bytes, input_image, best_match_img


def main():
    iface = gr.Interface(
        fn=run_prediction,
        inputs=[
            gr.Radio(["Danielle", "Felipe"], label="Choose a voice"),
            gr.Textbox(label="Enter the name of a celebrity that you find physically attractive")
        ],
        outputs=[
            gr.Audio(label="Output audio"),
            gr.Image(label="input image"),
            gr.Image(label="best match")
        ],
        title="Run Prediction",
        description="Take a photo and run prediction on it."
    )
    iface.launch(share=True)


if __name__ == "__main__":
    main()
