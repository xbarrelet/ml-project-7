import math
import os
import shutil
import time

import keras
import tensorflow as tf
from keras import layers, Sequential, Input
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.optimizers import Adam, AdamW
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from pandas import DataFrame
from plot_keras_history import plot_history

IMAGES_PATH = "resources/Images"
CROPPED_IMAGES_PATH = "resources/Cropped_Images2"
MODELS_PATH = "models/custom_model_advanced"
RESULTS_PATH = "results/custom_model_advanced"

# To optimize GPU memory consumption
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


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


def create_model(input_shape, labels_number, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Input(shape=input_shape),

        # Data augmentation layers
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def patch_embed(x, patch_size, embed_dim):
    patches = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)(x)
    b, h, w, c = patches.shape
    patches = layers.Reshape((h * w, c))(patches)
    return patches


def transformer_encoder(x, num_heads, mlp_units, dropout=0.1):
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attention_output = layers.Dropout(dropout)(attention_output)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

    # MLP
    mlp_output = layers.Dense(mlp_units, activation="gelu")(x1)
    mlp_output = layers.Dense(x.shape[-1])(mlp_output)
    mlp_output = layers.Dropout(dropout)(mlp_output)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + mlp_output)

    return x2


def create_cnn_vit_hybrid_model(input_shape, num_classes, patch_size=9, num_transformer_layers=2,
                                num_heads=4, mlp_units=64, embed_dim=128, dropout_rate=0.2, learning_rate=0.001):
    inputs = keras.Input(shape=input_shape)

    # FROM KERAS TUTO:
    #   learning_rate = 0.001
    # weight_decay = 0.0001
    # batch_size = 256
    # num_epochs = 10  # For real training, use num_epochs=100. 10 is a test value
    # image_size = 72  # We'll resize input images to this size
    # patch_size = 6  # Size of the patches to be extract from the input images
    # num_patches = (image_size // patch_size) ** 2
    # projection_dim = 64
    # num_heads = 4
    # transformer_units = [
    #     projection_dim * 2,
    #     projection_dim,
    # ]  # Size of the transformer layers
    # transformer_layers = 8
    # mlp_head_units = [
    #     2048,
    #     1024,
    # ]
    # Size of the dense layers of the final classifier

    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)

    # CNN feature extraction
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # # Patch embedding, the feature map is divided into patches and embedded for the VIT part
    x = patch_embed(x, patch_size, embed_dim)

    # # Positional embedding, gives the model information about the spatial relationships between patches
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_embed = layers.Embedding(input_dim=x.shape[1], output_dim=embed_dim)(positions)
    x = x + pos_embed

    # Transformer layers
    for _ in range(num_transformer_layers):
        x = transformer_encoder(x, num_heads, mlp_units, dropout_rate)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax")(x)

    model = keras.Model(inputs, outputs)

    optimizer = AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name, epoch=100, batch_size=16):
    checkpoint_path = f"{MODELS_PATH}/checkpoint_{model_name}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    fitting_start_time = time.time()
    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        batch_size=batch_size,
                        # epochs=1,
                        epochs=epoch,
                        callbacks=[checkpoint, reduce_lr, es],
                        verbose=1)
    fitting_time = time.time() - fitting_start_time

    model.load_weights(checkpoint_path)

    val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
    print(f"\nValidation Accuracy:{val_accuracy}.")

    test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
    print(f"\nTest Accuracy:{test_accuracy}.\n")

    plot_history(history, path=f"{RESULTS_PATH}/history_{model_name}.png")

    return {
        "fitting_time": fitting_time,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "model_name": model_name
    }


def display_results(results, hyperparameter_name):
    results_df = DataFrame(results)

    display_results_plot(results_df, hyperparameter_name, ["fitting_time"], "fitting_time")
    display_results_plot(results_df, hyperparameter_name, ["test_accuracy", "val_accuracy"], "accuracies",
                         ascending=False)
    display_results_plot(results_df, hyperparameter_name, ["test_loss", "val_loss"], "losses")


def display_results_plot(results, hyperparameter_name, metrics, metrics_name, ascending=True):
    results.sort_values(metrics[0], ascending=ascending, inplace=True)

    performance_plot = (results[metrics + ["model_name"]].plot(kind="bar", x="model_name", figsize=(15, 8), rot=0,
                                                               title=f"Results sorted by {metrics_name}"))
    performance_plot.title.set_size(20)
    performance_plot.set_xticks(range(0, len(results)))
    plt.xticks(rotation=90)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/{hyperparameter_name}_{metrics_name}_plot.png",
                                          bbox_inches='tight')
    # plt.show()
    plt.close()


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 32
    labels_number = 30

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.25,
                                data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.25,
                              data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, data_type=None)

    results = []
    for model_name in [
        "simple",
        "hybrid"
    ]:
        print(f"Starting training of {model_name} model.\n")

        if model_name == "simple":
            model = create_model(input_shape=image_size + (3,), labels_number=labels_number)
        else:
            model = create_cnn_vit_hybrid_model(input_shape=image_size + (3,), num_classes=labels_number)

        result = get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name)
        results.append(result)

        model.save(f"{MODELS_PATH}/model_{model_name}.keras")

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    display_results(sorted_results, "simple")

    print("Custom models learning script finished.\n")
