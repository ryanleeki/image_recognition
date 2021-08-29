import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image
from skimage.transform import resize
from tensorflow.python.client import device_lib

device_lib.list_local_devices()
tf.debugging.set_log_device_placement(False)

app = Flask(__name__)
image_folder = os.path.join("static", "images")
app.config["UPLOAD_FOLDER"] = image_folder


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():

    imagefile = request.files["imagefile"]
    image_path = os.path.join(image_folder, imagefile.filename)
    imagefile.save(image_path)

    from tensorflow.keras.preprocessing import image

    img = image.load_img(image_path, target_size=(160, 160))
    x = image.img_to_array(img)
    prediction = mdl.predict_image(x)

    return render_template("index.html", user_image=image_path, prediction_text=prediction)


def predict_image(image_path, model, class_names, visualize=1):

    container = []
    for file_name in os.listdir(image_path):
        container.append(np.array(Image.open(os.path.join(image_path, file_name))))

    container = [resize(img, [160, 160]) for img in container]
    # 5 160 160 3

    container = np.array(container)
    predictions = model.predict(container * 256)
    predictions = [np.argmax(pred) for pred in predictions]

    # np.stack(container).shape

    predictions = [class_names[pred] for pred in predictions]

    if visualize:
        plt.figure(figsize=(10, 10))

        for i in range(len(container)):
            _ = plt.subplot(1, 6, i + 1)
            plt.imshow((container[i] * 256).astype("uint8"))
            plt.title(predictions[i])
            plt.axis("off")

    return predictions


class Model_2:
    def __init__(self, model_location, class_name):
        self.model = tf.keras.models.load_model(model_location)
        self.class_name = class_name

    def predict_image(self, image):
        self.resized_image = resize(image, [160, 160])

        if self.resized_image.max() < 1.1:
            self.scaled_image = self.resized_image * 256
        else:
            self.scaled_image = self.resized_image
        # Needs to be taken out

        # If the first dimension of the image is not 1
        if self.scaled_image.shape[0] != 1:
            self.final_image = self.scaled_image[
                None,
            ]
        else:
            self.final_image = self.scaled_image
        pred_value = np.argmax(self.model.predict(self.final_image))
        pred_label = self.class_name[pred_value]

        return pred_label


if __name__ == "main":
    class_name = os.listdir(os.path.join("images", "train"))
    mdl = Model_2("trained_model", class_name)
    app.run()
