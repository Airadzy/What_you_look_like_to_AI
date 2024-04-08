import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
import pandas as pd
import keras
import tensorflow as tf

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)  # make sure its 2.15.0 !!!!


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


def run_prediction(img):
    feature_values = []
    img = np.expand_dims(img, axis=0)
    img = img / 255.
    home_dir = os.path.expanduser("~")
    downloads_path_os = os.path.join(home_dir, 'Downloads')
    h5_file_path = os.path.join(downloads_path_os, 'RESNET_MODEL.h5')
    model = load_model(h5_file_path)

    predictions = model.predict(img)
    # print("Predictions:", predictions)

    predictions = np.where(predictions > 0.4, 1, 0)
    connections = pd.DataFrame(predictions, columns=column_names[1:])
    pd.set_option("display.max_columns", None)

    for column in connections.columns:
        feature_name = column.replace('_', ' ')
        feature_value = 'yes' if connections[column].values[0] == 1 else 'no'
        feature_values.append(f"{feature_name}, {feature_value}")

    result_string = '. '.join(feature_values)
    print(result_string)
    return result_string
