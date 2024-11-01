import gc
import logging
import os
import shutil
import time

import keras
import tensorflow as tf
from keras import layers, Sequential
from keras import ops
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.optimizers import Adam, AdamW, RMSprop, SGD
from keras.src.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from pandas import DataFrame

IMAGES_PATH = "resources/Images"
CROPPED_IMAGES_PATH = "resources/cropped_Images_10_races"
MODELS_PATH = "models/custom_model_hyperoptimization"
MODEL_SAVE_PATH = f"{MODELS_PATH}/custom_model.keras"
RESULTS_PATH = "results/custom_model_hyperoptimization"

# To optimize GPU memory consumption
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


data_augmentation_layers = keras.Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(mode='horizontal'),
        layers.RandomContrast(factor=0.1),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomBrightness(factor=0.1)
    ],
    name="data_augmentation",
)


def remove_last_generated_models_and_results():
    shutil.rmtree(MODELS_PATH, ignore_errors=True)
    os.makedirs(MODELS_PATH)

    shutil.rmtree(RESULTS_PATH, ignore_errors=True)
    os.makedirs(RESULTS_PATH)


def start_logging():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='notebook2.log',
                        encoding='utf-8', level=logging.INFO)


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


def get_optimizer(optimizer, learning_rate):
    match optimizer:
        case "adam":
            return Adam(learning_rate=learning_rate)
        case "adamw":
            return AdamW(learning_rate=learning_rate)
        case "rmsprop":
            return RMSprop(learning_rate=learning_rate)
        case "sgd":
            return SGD(learning_rate=learning_rate)
        case "sgdn":
            return SGD(learning_rate=learning_rate, nesterov=True)
        case _:
            raise ValueError(f"Unknown optimizer:{optimizer}.")



def create_vit_model(input_shape, labels_number, patch_size=16, projection_dim=256, num_heads=8,
                     transformer_layers=8, mlp_head_units=256, learning_rate=0.001, optimizer="adam"):
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

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim // num_heads, dropout=0.1
        )(x1, x1)
        # Residual connections (via the Add layer) help preserve the original input to the layer.
        # This helps maintain gradients during backpropagation, making training more effective.
        x2 = layers.Add()([attention_output, patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation="gelu")(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        patches = layers.Add()([x3, x2])

    # Representation layers
    representation = layers.LayerNormalization(epsilon=1e-6)(patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    # Classify outputs
    features = layers.Dense(mlp_head_units, activation="gelu")(representation)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(labels_number, activation='softmax')(features)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=get_optimizer(optimizer, learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def display_results(results, hyperparameter_name):
    results_df = DataFrame(results)
    display_results_plot(results_df, hyperparameter_name, ["fitting_time"], "fitting_time")
    display_results_plot(results_df, hyperparameter_name, ["test_accuracy", "val_accuracy"], "accuracies",
                         ascending=False)
    display_results_plot(results_df, hyperparameter_name, ["test_loss", "val_loss"], "losses")


def display_results_plot(results, hyperparameter_name, metrics, metrics_name, ascending=True):
    results.sort_values(metrics[0], ascending=ascending, inplace=True)

    performance_plot = (results[metrics + ["hyperparameters_name"]]
                        .plot(kind="bar", x="hyperparameters_name", figsize=(15, 8), rot=0,
                              title=f"Results sorted by {metrics_name}"))
    performance_plot.title.set_size(20)
    performance_plot.set_xticks(range(0, len(results)))
    plt.xticks(rotation=90)
    performance_plot.set(xlabel=None)

    performance_plot.get_figure().savefig(f"{RESULTS_PATH}/{hyperparameter_name}_{metrics_name}_plot.png",
                                          bbox_inches='tight')
    # plt.show()
    plt.close()


def get_callbacks():
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, restore_best_weights=True)

    return [es, rlp]


def get_results_of_model(model, dataset_train, dataset_val, dataset_test, parameters, epoch=1000, batch_size=4):
    fitting_start_time = time.time()
    model.fit(dataset_train,
              validation_data=dataset_val,
              batch_size=batch_size,
              # epochs=2,
              epochs=epoch,
              callbacks=get_callbacks(),
              verbose=1)
    fitting_time = time.time() - fitting_start_time

    model.load_weights(CHECKPOINT_SAVE_PATH)

    val_loss, val_accuracy = model.evaluate(dataset_val, verbose=False)
    print(f"\nValidation Accuracy:{val_accuracy}.")

    test_loss, test_accuracy = model.evaluate(dataset_test, verbose=False)
    print(f"\nTest Accuracy:{test_accuracy}.\n")

    return {
        "hyperparameters_name": hyperparameters["name"],
        "fitting_time": fitting_time,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        **parameters
    }


def get_best_parameter(sorted_results, parameter_name):
    best_parameter = sorted_results[0][parameter_name]
    print(f"Best parameter:{parameter_name.replace("_", " ")} found:{best_parameter}.\n")
    return best_parameter


if __name__ == '__main__':
    print("Starting custom models learning script.\n")
    remove_last_generated_models_and_results()

    image_size = (224, 224)
    batch_size = 4
    labels_number = 20

    dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                                data_type='training')
    dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, validation_split=0.2,
                              data_type='validation')
    dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, batch_size, data_type=None)

    best_layers_parameters = {}

    # default_model = create_model(input_shape=image_size + (3,), labels_number=labels_number)
    # print(default_model.summary())

    # MODEL HYPEROPTIMIZATION
    results = []
    for hyperparameters in [
        {"name": "patch_size_4", "parameters": {"patch_size": 4}},
        {"name": "patch_size_8", "parameters": {"patch_size": 8}},
        {"name": "patch_size_12", "parameters": {"patch_size": 12}},
        {"name": "patch_size_16", "parameters": {"patch_size": 16}},
        {"name": "patch_size_25", "parameters": {"patch_size": 25}}
    ]:
        print(f"\nTesting now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["patch_size"] = get_best_parameter(sorted_results, "patch_size")
    display_results(sorted_results, "patch_size")

    results = []
    for hyperparameters in [
        {"name": "num_heads_1", "parameters": {"num_heads": 1}},
        {"name": "num_heads_2", "parameters": {"num_heads": 2}},
        {"name": "num_heads_4", "parameters": {"num_heads": 4}},
        {"name": "num_heads_6", "parameters": {"num_heads": 6}},
        {"name": "num_heads_8", "parameters": {"num_heads": 8}}
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      patch_size=best_layers_parameters["patch_size"],
                                      **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["num_heads"] = get_best_parameter(sorted_results, "num_heads")
    display_results(sorted_results, "num_heads")

    results = []
    for hyperparameters in [
        {"name": "transformer_layers_1", "parameters": {"transformer_layers": 1}},
        {"name": "transformer_layers_2", "parameters": {"transformer_layers": 2}},
        {"name": "transformer_layers_3", "parameters": {"transformer_layers": 3}},
        {"name": "transformer_layers_4", "parameters": {"transformer_layers": 4}},
        {"name": "transformer_layers_5", "parameters": {"transformer_layers": 5}},
        {"name": "transformer_layers_6", "parameters": {"transformer_layers": 6}},
        {"name": "transformer_layers_8", "parameters": {"transformer_layers": 8}}
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      patch_size=best_layers_parameters["patch_size"],
                                      num_heads=best_layers_parameters["num_heads"],
                                      **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["transformer_layers"] = get_best_parameter(sorted_results, "transformer_layers")
    display_results(sorted_results, "transformer_layers")

    results = []
    for hyperparameters in [
        {"name": "mlp_head_units_64", "parameters": {"mlp_head_units": 64}},
        {"name": "mlp_head_units_128", "parameters": {"mlp_head_units": 128}},
        {"name": "mlp_head_units_256", "parameters": {"mlp_head_units": 256}},
        {"name": "mlp_head_units_512", "parameters": {"mlp_head_units": 512}},
        {"name": "mlp_head_units_1024", "parameters": {"mlp_head_units": 1024}}
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      patch_size=best_layers_parameters["patch_size"],
                                      num_heads=best_layers_parameters["num_heads"],
                                      transformer_layers=best_layers_parameters["transformer_layers"],
                                      **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["mlp_head_units"] = get_best_parameter(sorted_results, "mlp_head_units")
    display_results(sorted_results, "mlp_head_units")


    results = []
    for hyperparameters in [
        {"name": "projection_dim_16", "parameters": {"projection_dim": 16}},
        {"name": "projection_dim_32", "parameters": {"projection_dim": 32}},
        {"name": "projection_dim_64", "parameters": {"projection_dim": 64}},
        {"name": "projection_dim_128", "parameters": {"projection_dim": 128}},
        {"name": "projection_dim_256", "parameters": {"projection_dim": 256}}
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      patch_size=best_layers_parameters["patch_size"],
                                      num_heads=best_layers_parameters["num_heads"],
                                      transformer_layers=best_layers_parameters["transformer_layers"],
                                      mlp_head_units=best_layers_parameters["mlp_head_units"],
                                      **hyperparameters["parameters"])
        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["projection_dim"] = get_best_parameter(sorted_results, "projection_dim")
    display_results(sorted_results, "projection_dim")


    # COMPILATION HYPEROPTIMIZATION
    results = []
    for hyperparameters in [
        {"name": "rmsprop_optimizer", "parameters": {"optimizer": "rmsprop"}},
        {"name": "adam_optimizer", "parameters": {"optimizer": "adam"}},
        {"name": "adamw_optimizer", "parameters": {"optimizer": "adamw"}},
        {"name": "sgd_optimizer", "parameters": {"optimizer": "sgd"}},
        {"name": "sgd_nesterov_optimizer", "parameters": {"optimizer": "sgdn"}},
    ]:
        print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
        model = create_vit_model(input_shape=image_size + (3,), labels_number=labels_number,
                                      patch_size=best_layers_parameters["patch_size"],
                                      projection_dim=best_layers_parameters["projection_dim"],
                                      num_heads=best_layers_parameters["num_heads"],
                                      transformer_layers=best_layers_parameters["transformer_layers"],
                                      mlp_head_units=best_layers_parameters["mlp_head_units"],
                                      **hyperparameters["parameters"])

        results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
                                            hyperparameters["parameters"]))


    sorted_results = sorted(results, key=lambda x: x["val_loss"])
    best_layers_parameters["optimizer"] = get_best_parameter(sorted_results, "optimizer")
    display_results(sorted_results, "optimizer")


    # EXECUTION HYPEROPTIMIZATION
    # results = []
    # for hyperparameters in [
    #     {"name": "batch_size_4", "parameters": {"batch_size": 4}},
    #     {"name": "batch_size_8", "parameters": {"batch_size": 8}},
    #     {"name": "batch_size_16", "parameters": {"batch_size": 16}},
    #     {"name": "batch_size_32", "parameters": {"batch_size": 32}},
    # ]:
    #     print(f"Testing now the parameters:{hyperparameters["parameters"]}.\n")
    #
    #     model = create_simplified_vit(input_shape=image_size + (3,), labels_number=labels_number,
    #                                   patch_size=best_layers_parameters["patch_size"],
    #                                   projection_dim=best_layers_parameters["projection_dim"],
    #                                   num_heads=best_layers_parameters["num_heads"],
    #                                   transformer_layers=best_layers_parameters["transformer_layers"],
    #                                   mlp_head_units=best_layers_parameters["mlp_head_units"],
    #                                   optimizer=best_layers_parameters["optimizer"],
    #                                   learning_rate=best_layers_parameters["learning_rate"])
    #
    #     new_batch_size = hyperparameters["parameters"]["batch_size"]
    #     dataset_train = get_dataset(CROPPED_IMAGES_PATH, image_size, new_batch_size, validation_split=0.25,
    #                                 data_type='training')
    #     dataset_val = get_dataset(CROPPED_IMAGES_PATH, image_size, new_batch_size, validation_split=0.25,
    #                               data_type='validation')
    #     dataset_test = get_dataset(CROPPED_IMAGES_PATH, image_size, new_batch_size, data_type=None)
    #
    #     results.append(get_results_of_model(model, dataset_train, dataset_val, dataset_test,
    #                                         hyperparameters["parameters"],
    #                                         epoch=best_layers_parameters["epoch"],
    #                                         batch_size=new_batch_size))


    # sorted_results = sorted(results, key=lambda x: x["val_loss"])
    # best_layers_parameters["batch_size"] = get_best_parameter(sorted_results, "batch_size")
    # display_results(sorted_results, "batch_size")

    print(f"Hyperoptimization now done. Best hyperparameters found:{best_layers_parameters}.\n")
    print("Custom models learning script finished.\n")
