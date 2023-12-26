from tensorflow import keras
import cv2
import cvlib as cv
import numpy as np
import base64
import io
from PIL import Image

def FER(imgdata):
    #emotion = ["happy", "surprise", "angry", "sad", "neutral"] 
    model = keras.models.load_model('data/model.h5')

    # base64 to image
    image = Image.open(io.BytesIO(imgdata))
    image = np.array(image)

    #detect face and crop
    faces, conf = cv.detect_face(image)
    for (x, y, x2, y2) in faces:
        cropped = image[y:y2, x:x2]
    resize = cv2.resize(cropped, (96,96))
    img = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    # image normalization
    img = img / 255            
    img = img.reshape(96, 96, 1)
    img = np.expand_dims(img, axis=0)

    #predict
    pred = model.predict(img, verbose=0)

    return pred







