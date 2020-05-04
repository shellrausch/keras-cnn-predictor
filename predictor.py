import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model_name = os.environ['MODEL_NAME']
model_path = os.path.join("models", model_name)
predict_images_path = "tmp"
valid_img_extensions = [".png", ".jpg", ".jpeg", ".bmp"]


def predict_img(img, model):
    img = img.resize((224, 224))
    img = img.convert("RGB")

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    return classes


model = load_model(model_path)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

print("Start predicting")

for filename in os.listdir(predict_images_path):
    path = os.path.join(predict_images_path, filename)
    filename, ext = os.path.splitext(path)

    # no file extension present
    if not ext:
        continue

    # has valid image extension
    if ext.lower() not in valid_img_extensions:
        continue

    img = image.load_img(path)
    preds = predict_img(img, model)

    print(str(preds[0][0]), path)

print("Prediction finished")
