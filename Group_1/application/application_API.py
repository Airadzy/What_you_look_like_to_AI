# import numpy as np
# import cv2
# import streamlit as st
# from tensorflow import keras
# from keras.models import model_from_json
# from keras.preprocessing.image import img_to_array
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# load model
# emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
# load json and create model
# json_file = open('emotion_model1.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)

# load weights into new model
# classifier.load_weights("emotion_model1.h5")

# load face
# try:
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# except Exception:
#     st.write("Error loading cascade classifiers")


# class Faceemotion(VideoTransformerBase):
#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")

# image gray
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(
#     image=img_gray, scaleFactor=1.3, minNeighbors=5)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img=img, pt1=(x, y), pt2=(
#         x + w, y + h), color=(255, 0, 0), thickness=2)
#     roi_gray = img_gray[y:y + h, x:x + w]
#     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#     if np.sum([roi_gray]) != 0:
#         roi = roi_gray.astype('float') / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)
#         prediction = classifier.predict(roi)[0]
#         maxindex = int(np.argmax(prediction))
#         finalout = emotion_dict[maxindex]
#         output = str(finalout)
#     label_position = (x, y)
#     cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# return img

import streamlit as st
import time  # Import the time module for time-related operations
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import json
import requests
from keras.models import load_model
import numpy as np
import cv2
import pandas as pd

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class Faceemotion(VideoTransformerBase):
    def __init__(self):
        self.last_saved_time = time.time()  # Initialize the last saved time
        self.call_test = True  # Initialize flag to control calling test()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Check if 2 seconds have passed since the last frame was saved
        if time.time() - self.last_saved_time >= 2:
            self.last_saved_time = time.time()  # Update the last saved time
            # Save the frame to a variable
            saved_frame = img
            print(saved_frame.shape)
            if self.call_test == True:
                prediction_result = self.prediction_img(saved_frame)
                result = self.text_generator(prediction_result)
                self.voice_generator(result)  # Call the test function
                self.call_test = False  # Set flag to False to stop calling test()

        return img
    def prediction_img(self,frame):
        model = load_model('model_epochRESNET_08_(1).h5')

        # resize
        resized_image = cv2.resize(frame, (218, 178))  # Change dimensions as needed
        image_with_dim = np.expand_dims(resized_image, axis=0)

        predictions = model.predict(image_with_dim)
        predicted_labels = np.where(predictions > 0.5, 1, 0)

        column_names = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                        'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                        'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                        'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                        'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                        'Wearing_Necktie', 'Young']

        connections = pd.DataFrame(predicted_labels, columns=column_names[1:])
        pd.set_option("display.max_columns", None)

        attribute = []
        for feat in connections:
            if connections[feat].values == 1:
                attribute.append(feat)

        return attribute

    def text_generator(self,prediction):
        # key API
        final_attribute=" ".join(prediction)
        headers = {
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZTZhOGQxZmMtZjc1Ny00OTVlLTgzNzgtMDJkZjExNjhhNTg3IiwidHlwZSI6ImFwaV90b2tlbiJ9.t6D9SSDFgLrHy1GmywPam2MHTrN2qcEIZWsYO3KZEbk"}

        url = "https://api.edenai.run/v2/text/generation"

        # create the prompt attributes should take from the prediction
        intro = " create a text that describe a personn with the following attribute in 6 words : "
        attributes = final_attribute
        prompt = intro + attributes

        # create the question
        payload = {"providers": "openai,cohere",
                   "text": prompt,
                   "temperature": 0.2,
                   "max_tokens": 250
                   }
        response = requests.post(url, json=payload, headers=headers)

        # print the answer
        result = json.loads(response.text)
        description = result['openai']['generated_text']
        return description

    # creating the voice
    def voice_generator(self, description):
        CHUNK_SIZE = 1024
        url = "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": "642a30a506ea5fc233f1997618b4cb05"
        }

        data = {
            "text": description,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }

        response = requests.post(url, json=data, headers=headers)
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        print("launch the audio")
        #             launch the audio

def listen_audio(audio_file_path):
    print("in listen audio")
    st.audio("output.mp3", format='audio/mp3', start_time=0)


def main():
    # Face Analysis Application #
    st.title("Real Time Face Description")
    st.header("Welcome to our application! Click on START button and let the magic operate!")
    st.write("Webcam Live Feed")

    # Start the webcam feed
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=Faceemotion)

    st.title("Audio Player")
    if st.button('listen Audio'):
        listen_audio("output.mp3")

if __name__ == "__main__":
    main()
