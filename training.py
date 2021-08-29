import argparse
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Training
# Padding Convet
def import_images(path, IMG_SIZE=(160, 160), visualize=0):
    train_dir = os.path.join(path, "images\\train")
    validation_dir = os.path.join(path, "images\\validation")

    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    with tf.device("/gpu:0"):
        train_dataset = image_dataset_from_directory(
            train_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
        )

        validation_dataset = image_dataset_from_directory(
            validation_dir, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
        )

        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)

    class_names = train_dataset.class_names

    if visualize == 1:

        plt.figure(figsize=(17, 10))
        for images, labels in train_dataset.take(1):
            for i in range(BATCH_SIZE):
                _ = plt.subplot(4, 8, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

    print("Number of validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
    print("Number of test batches: %d" % tf.data.experimental.cardinality(test_dataset))

    return train_dataset, validation_dataset, val_batches, test_dataset, IMG_SIZE


class Model:
    def __init__(self, train_dataset, validation_dataset, IMG_SIZE):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.3, 0)),
            ]
        )
        self.preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        self.rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 127.5, offset=-1)
        self.IMG_SHAPE = IMG_SIZE + (3,)

    def train(
        self,
        base_learning_rate=0.0001,
        initial_epochs=10,
        fine_tune_epochs=20,
        fine_tune_at=50,
        folder_name="trained_model",
    ):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.IMG_SHAPE, include_top=False, weights="imagenet"
        )
        base_model.trainable = False

        image_batch, label_batch = next(iter(self.train_dataset))
        # feature_batch = base_model(image_batch)

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # feature_batch_average = global_average_layer(feature_batch)

        prediction_layer = tf.keras.layers.Dense(5, activation="softmax")
        # prediction_batch = prediction_layer(feature_batch_average)

        with tf.device("/gpu:0"):
            inputs = tf.keras.Input(shape=(160, 160, 3))
            x = self.data_augmentation(inputs)
            x = self.preprocess_input(x)
            x = base_model(x, training=False)
            x = global_average_layer(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = prediction_layer(x)
            model = tf.keras.Model(inputs, outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )
            # breakpoint()
            self.history = model.fit(self.train_dataset, epochs=initial_epochs, validation_data=self.validation_dataset)

            base_model.trainable = True

            # Fine-tune from this layer onwards
            # fine_tune_at = 50

            # Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False

            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
                metrics=["accuracy"],
            )

            total_epochs = initial_epochs + fine_tune_epochs

            self.history_finetune = model.fit(
                self.train_dataset,
                epochs=total_epochs,
                initial_epoch=self.history.epoch[-1],
                validation_data=self.validation_dataset,
            )

            model.save(folder_name)

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Please enter folder name for a saved model.")
    parser.add_argument("--folder_name", type=str, help="Folder name for a trained model", default="trained_model_2")
    args = parser.parse_args()
    train_dataset, validation_dataset, val_batches, test_dataset, IMG_SIZE = import_images("")
    model = Model(train_dataset, validation_dataset, IMG_SIZE)
    model.train(initial_epochs=1, fine_tune_epochs=1, folder_name=args.folder_name)
