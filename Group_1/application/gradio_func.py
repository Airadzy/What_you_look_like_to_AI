import gradio as gr
import cv2
from Group_1.application import prediction
from Group_1.application.narrator import create_audio
import return_images
import os
import json

# Load configuration from a JSON file
with open('config.json') as config_file:
    config = json.load(config_file)


def fetch_initial_match(celebrity_names, matches_generator, first_match_shown):
    image_path = return_images.download_first_image(celebrity_names, config["google_API_key"], config["google_cx"])
    if not image_path or not os.path.exists(image_path):
        return None, "Image not available", "No image found for the celebrity", matches_generator, False

    img = cv2.imread(image_path)
    if img is None:
        return None, "Failed to load image", "Error loading celebrity image", matches_generator, False
    resized_img = cv2.resize(img, (218, 178), interpolation=cv2.INTER_AREA)
    cv2.imwrite("current_image.jpg", resized_img)
    prediction_string, connections = prediction.run_prediction(resized_img)
    audio_bytes = create_audio(prediction_string)

    matches_generator = iter(return_images.find_best_match(connections, config["csv_path"]))
    next_match = next(matches_generator)
    best_match_path = os.path.join(config["images_path"], f"{next_match}")
    best_match_img = cv2.imread(best_match_path)
    if best_match_img is None:
        return None, "Failed to load image", "Match image not available", matches_generator, False

    best_match_img = cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB)
    return audio_bytes, best_match_img, "Initial match loaded. Swipe to see more.", matches_generator, True


def handle_swipe(action, matches_generator, first_match_shown):
    if not first_match_shown or matches_generator is None:
        return None, f"Load an initial match first. Action was: {action}", matches_generator, first_match_shown
    try:
        next_match = next(matches_generator)
        best_match_path = os.path.join(config["images_path"], f"{next_match}")
        best_match_img = cv2.imread(best_match_path)
        best_match_img = cv2.cvtColor(best_match_img, cv2.COLOR_BGR2RGB)
        if action == "SWIPING LEFT":
            return best_match_img, "😔😔😔NO MATCH😔😔😔. Swipe again or submit new name.", matches_generator, first_match_shown
        elif action == "SWIPING RIGHT":
            return best_match_img, "❤️️❤️️❤️️IT'S A MATCH❤️️❤️️❤️️. How about this one?", matches_generator, first_match_shown
    except StopIteration:
        return None, "No more matches available.", matches_generator, first_match_shown


def main():
    with gr.Blocks() as app:
        with gr.Row():
            celebrity_input = gr.Textbox(label="Enter the name of a celebrity that you find physically attractive and press submit")
            submit_button = gr.Button("Submit")
        image_output = image_output = gr.Image(label="Possible match",height=218,width=178)
        with gr.Row():
            swipe_left_button = gr.Button("SWIPE LEFT")
            swipe_right_button = gr.Button("SWIPE RIGHT")
        response_output = gr.Text(label="Match Response")
        audio_output = gr.Audio(label="Output audio")

        matches_generator = gr.State(None)
        first_match_shown = gr.State(False)

        submit_button.click(fetch_initial_match, inputs=[celebrity_input, matches_generator, first_match_shown],
                            outputs=[audio_output, image_output, response_output, matches_generator, first_match_shown])

        swipe_left_button.click(
            handle_swipe,
            inputs=[gr.Text("SWIPING LEFT"), matches_generator, first_match_shown],
            outputs=[image_output, response_output, matches_generator, first_match_shown]
        )
        swipe_right_button.click(
            handle_swipe,
            inputs=[gr.Text("SWIPING RIGHT"), matches_generator, first_match_shown],
            outputs=[image_output, response_output, matches_generator, first_match_shown]
        )

    app.launch(share=True)


if __name__ == "__main__":
    main()
