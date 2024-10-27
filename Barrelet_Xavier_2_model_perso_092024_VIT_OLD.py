import glob
import os
import shutil
import time
from random import randint

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import layers, Sequential
from keras.src.applications.efficientnet_v2 import EfficientNetV2L
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from pandas import DataFrame
from plot_keras_history import plot_history
from skimage.filters import gaussian
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# from transformers import TFViTModel
# from transformers import ViTFeatureExtractor

CROPPED_IMAGES_PATH = "resources/Cropped_Images2"
MODELS_PATH = "models/custom_model_vit"
RESULTS_PATH = "results/custom_model_vit"


data_augmentation_layers = keras.Sequential(
    [
        layers.RandomRotation(factor=0.15, input_shape=(224, 224, 3)),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode='horizontal'),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomBrightness(factor=0.1)
    ],
    name="data_augmentation",
)


class RISE:
    """
    Generate heatmap explanations for image classifiers using the RISE methodology by Petsiuk et al.
    (reference: https://arxiv.org/abs/1806.07421)
    Generate N binary masks of initial size s by s, which are then upsampled and applied to an image.
    Elements in the initial arrays are set to 1 with probability p1. Else, they are set to 0.
    The final heatmap is generated as a linear combination of the masks.
    The weights are obtained from the softmax probabilities predicted by the base model on the masked images
    """

    def __init__(self):

        self.model = None
        self.input_size = None
        self.masks = None

    def generate_masks(self, N, s, p1):

        """
        Generate a distribution of random binary masks.

        Args:
            N: Number of masks.
            s: Size of mask before upsampling.
            p1: Probability of setting element value to 1 in the initial mask.
            verbose: Verbose level for the model prediction step.
            batch_size: Batch size for predictions.

        Returns:
            masks: The distribution of upsampled masks.
        """

        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        masks = np.empty((N, *self.input_size))

        for i in range(N):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                    anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        masks = masks.reshape(-1, *self.input_size, 1)
        return masks

    def explain(
            self,
            inp,
            model,
            preprocessing_fn=None,
            masks_user=None,
            N=2000,
            s=8,
            p1=0.5,
            verbose=0,
            batch_size=100,
            mode=None
    ):
        """
        Generate the explanation heatmaps for all classes.

        Args:
            model: The image classifier. Typically expects a Tensorflow 2.0/Keras model or equivalent class.
            inp: The image to be explained. Expected to be in the shape used by the model, without any color
            normalization or futher preprocessing applied. Ideally the any color preprocessing is included
            within the model class/function.
            preprocessing_fn: Not implemented yet. For now preprocessing should ideally be included within the model.
            masks_user: This function calls another function to generate a mask distribution. However a user generated
            distribution of masks can be passed with this argument.
            N: Number of masks.
            s: Size of mask before upsampling.
            p1: Probability of setting element value to 1 in the initial mask.
            verbose: Verbose level for the model prediction step.
            batch_size: Batch size for predictions.
            mode (experimental): Alternative perturbation modes instead of the simple black gradation mask. 'blur'
            is a Gaussian blur, 'noise' is colored noise and 'noise_bw' is black and white noise. If None will return
            the regular black gradation perturbations. Default is None.

        Returns:
            sal: Explanation heatmaps for all classes. For a given class_id, the heatmap can be access
            with sal[class_id].
            masks: The distribution of masks used for generating the set of heatmaps.
        """
        self.model = model
        self.input_size = model.input_shape[1:3]

        if masks_user == None:
            self.masks = self.generate_masks(N, s, p1)
        else:
            self.masks = masks_user  # In case the user wants to pass some custom numpy array of masks.

        # Make sure multiplication is being done for correct axes

        image = inp
        fudged_image = image.copy()

        if mode == 'blur':  # Gaussian blur
            fudged_image = gaussian(fudged_image, sigma=4, multichannel=True, preserve_range=True)

        elif mode == 'noise':  # Colored noise
            fudged_image = np.random.normal(255 / 2, 255 / 9, size=fudged_image.shape).astype('int')

        elif mode == 'noise_bw':  # Grayscale noise
            fudged_image = np.random.normal(255 / 2, 255 / 5, size=(fudged_image.shape[:2])).astype('int')
            fudged_image = np.stack((fudged_image,) * 3, axis=-1)

        else:
            fudged_image = np.zeros(image.shape)  # Regular perturbation with a black gradation

        preds = []

        # Doing these matrix multiplications between the masks and the image can quickly eat up memory.
        # So we multiply the image with one batch of masks at a time and later append the predictions.

        if (verbose):
            print('Using batch size: ', batch_size, flush=True)

        for i in (tqdm(range(0, N, batch_size)) if verbose else range(0, N, batch_size)):
            masks_batch = self.masks[i:min(i + batch_size, N)]
            masked = image * masks_batch + fudged_image * (1 - masks_batch)

            to_append = model.predict(masked)

            preds.append(to_append)

        preds = np.vstack(preds)

        sal = preds.T.dot(self.masks.reshape(N, -1)).reshape(-1, *self.input_size)
        sal = sal / N / p1

        return sal, self.masks


def remove_last_generated_models_and_results():
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    os.makedirs(MODELS_PATH)

    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.makedirs(RESULTS_PATH)


def get_dataset(path, image_size, batch_size, validation_split=0.0, data_type=None):
    return image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        batch_size=batch_size,
        image_size=image_size,
        seed=42,
        validation_split=validation_split,
        subset=data_type
    )


def create_cnn_model(input_shape, labels_number, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        keras.Input(shape=input_shape),

        data_augmentation_layers,

        # CNN layers
        layers.Conv2D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # END LAYERS
        layers.GlobalAveragePooling2D(),
        layers.Dropout(dropout_rate),
        layers.Dense(labels_number, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_complex_cnn_model(input_shape, labels_number, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        keras.Input(shape=input_shape),

        data_augmentation_layers,

        # Block 1
        layers.Conv2D(64, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 2
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 3
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 4
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Final layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(labels_number, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def create_vit_model(input_shape, labels_number, patch_size=16, projection_dim=256, num_heads=8,
                     transformer_layers=8, mlp_head_units=256, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape)

    inputs = data_augmentation_layers(inputs)

    # Create patches
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
    patch_dims = patches.shape[1] * patches.shape[2]
    patches = layers.Reshape((patch_dims, projection_dim))(patches)

    # Add positional embeddings
    positions = tf.range(start=0, limit=patch_dims, delta=1)
    pos_embedding = layers.Embedding(input_dim=patch_dims, output_dim=projection_dim)(positions)
    patches += pos_embedding

    # Create multiple layers of the Transformer block
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    # Classify outputs
    features = layers.Dense(mlp_head_units, activation="gelu")(representation)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(labels_number, activation='softmax')(features)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name, epoch=1000, batch_size=32):
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)

    fitting_start_time = time.time()
    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        batch_size=batch_size,
                        # epochs=2,
                        epochs=epoch,
                        callbacks=[rlp, es],
                        verbose=1)
    fitting_time = time.time() - fitting_start_time

    val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
    print(f"\nValidation Accuracy:{val_accuracy}.")
    test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
    print(f"\nTest Accuracy:{test_accuracy}.\n")

    plot_history(history, path=f"{RESULTS_PATH}/history_{model_name}.png")
    # show_history(history)

    return {
        "fitting_time": fitting_time,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "model_name": model_name
    }


def display_results(results):
    results_df = DataFrame(results)

    display_results_plot(results_df, ["fitting_time"], "fitting_time")
    display_results_plot(results_df, ["test_accuracy", "val_accuracy"], "accuracies",
                         ascending=False)
    display_results_plot(results_df, ["test_loss", "val_loss"], "losses")


def display_results_plot(results, metrics, metrics_name, ascending=True):
    results.sort_values(metrics[0], ascending=ascending, inplace=True)

    performance_plot = (results[metrics + ["model_name"]].plot(kind="bar", x="model_name", figsize=(15, 8), rot=0,
                                                               title=f"Results sorted by {metrics_name}"))
    performance_plot.title.set_size(20)
    performance_plot.set_xticks(range(0, len(results)))
    plt.xticks(rotation=90)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/{metrics_name}_plot.png", bbox_inches='tight')
    # plt.show()
    plt.close()


def load_images():
    images_df = DataFrame()

    all_images = list(glob.glob(f"{CROPPED_IMAGES_PATH}/*/*.jpg"))
    images_df["image_path"] = all_images

    images_df["label_name"] = images_df["image_path"].apply(lambda path: path.split("/")[-2].lower())

    labels = [f.path.split("/")[-1].lower() for f in os.scandir(CROPPED_IMAGES_PATH) if f.is_dir()]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    images_df["label"] = label_encoder.transform(images_df["label_name"])

    return images_df


def explain_images(model, model_name):
    images_df = load_images()
    explainer = RISE()

    for index in range(4):
        row = images_df.iloc[randint(0, len(images_df) - 1)]

        image = Image.open(row['image_path'])
        image = np.array(image.resize((224, 224)))

        heatmaps, masks = explainer.explain(image, model)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax2.imshow(heatmaps[row['label']], cmap='jet')
        ax2.imshow(image, alpha=0.5)

        plt.axis('off')
        fig.savefig(f"{RESULTS_PATH}/explanation_{model_name}_{index + 1}.png", bbox_inches='tight')
        # plt.show()
        plt.close()


def create_pre_trained_cnn_model(input_shape, labels_number, learning_rate=0.001):
    base_model = EfficientNetV2L(include_top=False, weights="imagenet", input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        data_augmentation_layers,

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(labels_number, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def preprocess_image_for_pretrained_vit_model(image, feature_extractor):
    """Preprocess image according to ViT requirements"""
    try:
        # Ensure image is in RGB format
        if image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(image)

        # Resize image to 224x224 (ViT's expected input size)
        image = tf.image.resize(image, (224, 224))

        # Convert to float32 and normalize
        image = tf.cast(image, tf.float32) / 255.0

        # Expand dimensions to create batch
        image = tf.expand_dims(image, 0)

        # Preprocess using the feature extractor
        inputs = feature_extractor(images=image.numpy(), return_tensors="tf")

        return inputs['pixel_values'][0]
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None


def create_pre_trained_vit_model(input_shape, labels_number, learning_rate=0.001):
    model_name = "google/vit-base-patch16-224"

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    vit_model = TFViTModel.from_pretrained(model_name)

    for layer in vit_model.layers:
        layer.trainable = False

    inputs = layers.Input(shape=input_shape)

    # Preprocess the input using the feature extractor
    x = keras.layers.Lambda(lambda img: feature_extractor(images=img.numpy(), return_tensors="tf")['pixel_values'])(
        inputs)

    # Use the ViT model
    x = vit_model(x)[0]  # Get the output from the ViT model (the last hidden state)

    # Add a classification head
    x = layers.GlobalAveragePooling1D()(x)  # Use global average pooling to reduce dimensions
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(labels_number, activation='softmax')(x)

    model = keras.models.Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_model(model_name, labels_number):
    if model_name == "cnn":
        return create_cnn_model(input_shape=image_size + (3,), labels_number=labels_number)
    elif model_name == "complex-cnn":
        return create_complex_cnn_model(input_shape=image_size + (3,), labels_number=labels_number)
    elif model_name == "vit":
        return create_vit_model(input_shape=image_size + (3,), labels_number=labels_number)
    elif model_name == "pre-trained-cnn":
        return create_pre_trained_cnn_model(input_shape=image_size + (3,), labels_number=labels_number)
    elif model_name == "pre-trained-vit":
        return create_pre_trained_vit_model(input_shape=image_size + (3,), labels_number=labels_number)


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 32
    labels_number = 5

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                                data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                              data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, data_type=None)

    results = []
    for model_name in [
        "cnn",
        "complex-cnn",
        "vit",
        # "pre-trained-cnn",
        # "pre-trained-vit"
    ]:
        print(f"\nStarting training of {model_name} model.\n")

        # TODO: Try only the custom models on 3, 10 and all races?
        model = get_model(model_name, labels_number)

        result = get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name,
                                      batch_size=batch_size, epoch=100)
        results.append(result)

        model.save(f"{MODELS_PATH}/model_{model_name}.keras")

        explain_images(model, model_name)

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    display_results(sorted_results)

    print("Custom models learning script finished.\n")
