import os
import shutil
import time

import keras
from keras import layers, Sequential, Input
from keras import ops
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.optimizers import Adam, AdamW
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from pandas import DataFrame
from plot_keras_history import plot_history
import tensorflow as tf

CROPPED_IMAGES_PATH = "resources/Cropped_Images"
MODELS_PATH = "models/custom_model_vit"
RESULTS_PATH = "results/custom_model_vit"

# To optimize GPU memory consumption
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

data_augmentation_layers = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        # layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)


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
        Input(shape=input_shape),

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


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


# The PatchEncoder layer will linearly transform a patch by projecting it into a vector of size projection_dim.
# In addition, it adds a learnable position embedding to the projected vector.
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


def create_vit_model(input_shape, num_classes, image_size=224, patch_size=16, projection_dim=64, num_heads=4,
                     transformer_layers=8, mlp_first_head_units=2048, learning_rate=0.001, weight_decay=0.0001):
    num_patches = (image_size // patch_size) ** 2
    # Size of the transformer layers
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]
    # Size of the dense layers of the final classifier
    mlp_head_units = [
        mlp_first_head_units,
        int(mlp_first_head_units/2),
    ]

    inputs = keras.Input(shape=input_shape)

    augmented = data_augmentation_layers(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    # Classify outputs.
    outputs = layers.Dense(num_classes, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=AdamW(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            # keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    return model


def create_simplified_vit(input_shape, num_classes, patch_size=16, projection_dim=64, num_heads=4,
                          transformer_layers=4, mlp_head_units=256, learning_rate=0.001):
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
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(patches)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
        )(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip connection 2
        patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    # Classify outputs
    features = layers.Dense(mlp_head_units, activation="gelu")(representation)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(num_classes, activation='softmax')(features)

    # Create the Keras model
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=AdamW(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            # keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    return model

def get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name, epoch=1000, batch_size=32):
    checkpoint_path = f"{MODELS_PATH}/checkpoint_{model_name}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    fitting_start_time = time.time()
    history = model.fit(dataset_train,
                        validation_data=dataset_val,
                        batch_size=batch_size,
                        # epochs=2,
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


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 32
    labels_number = 120

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                                data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                              data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, data_type=None)

    results = []
    for model_name in [
        "simple",
        "vit",
        # "vit_advanced"
    ]:
        print(f"Starting training of {model_name} model.\n")

        if model_name == "simple":
            model = create_cnn_model(input_shape=image_size + (3,), labels_number=labels_number)
        elif model_name == "vit":
            model = create_simplified_vit(input_shape=image_size + (3,), num_classes=labels_number)
        else:
            model = create_vit_model(input_shape=image_size + (3,), num_classes=labels_number)

        result = get_results_of_model(model, dataset_train, dataset_val, dataset_test, model_name,
                                      batch_size=batch_size)
        results.append(result)

        model.save(f"{MODELS_PATH}/model_{model_name}.keras")

    sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
    display_results(sorted_results, "simple")

    print("Custom models learning script finished.\n")
