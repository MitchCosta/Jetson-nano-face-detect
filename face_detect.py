import cv2

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
# import tensorflow as tf
import PIL

# import cv2

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

print("Imports done")

# /home/miguel/Documents/Deeplearning/models

from tensorflow.keras.models import model_from_json


def gstreamer_pipeline(
        capture_width=3264,
        capture_height=2464,
        display_width=820,
        display_height=820,
        framerate=21,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate, s
                flip_method,
                display_width,
                display_height,
            )
    )


json_file = open('/home/miguel/Documents/Deeplearning/models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('/home/miguel/Documents/Deeplearning/models/model.h5')

print(model.inputs)
print(model.outputs)

FRmodel = model


# tf.keras.backend.set_image_data_format('channels_last')
def img_to_encoding(img, model):
    # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    # img = image_path.resize((160, 160))
    # img = tf.keras.preprocessing.image.resize(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)

    cv2.imshow("Face to feed", img)

    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


print("Starting database generation...")


def face_detect():
    encoding_new = 0
    encoding_stored = 0

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        cv2.namedWindow("Face Detect", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("Face Detect", 0) >= 0:
            ret, img = cap.read()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            cv2.imshow("Face Detect", img)
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break


            img = cv2.resize(img, (160, 160))
            encoding_new = img_to_encoding(img, model)

            # print("keycode", keyCode)
            # a -> 97
            # c -> 99

            if keyCode == 99:
                print("Picture stored   ")
                encoding_stored = encoding_new

            print("Distance = ", np.linalg.norm(encoding_new - encoding_stored))

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


face_detect()
