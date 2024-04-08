import cv2
import time
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def take_photo():
    height = 178
    width = 218
    filename = "frame.jpg"
    frames_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    time.sleep(2)

    ret, frame = cap.read()
    if ret:

        resized_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Save the frame as an image file
        print("📸 Say cheese! Saving frame.")
        path = os.path.join(frames_dir, filename)
        cv2.imwrite(path, resized_img)
    else:
        print("Failed to capture image")

        # Wait for 2 seconds
        time.sleep(2)

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return resized_img
