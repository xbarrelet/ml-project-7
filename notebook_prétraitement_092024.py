import glob
import os
from xml.etree import ElementTree

import cv2
import plotly.express as px
from PIL import Image
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

IMAGES_PATH = "resources/Images"
CROPPED_IMAGES_PATH = "resources/Cropped_Images"
# CROPPED_IMAGES_PATH = "resources/cropped_Images_10_races"
MODELS_PATH = "models"
PLOTLY_PATH = "results/plotly"



def extract_information_from_annotations(image_path):
    annotation_path = image_path.replace("Images", "Annotation").replace(".jpg", "")

    tree = ElementTree.parse(annotation_path)
    x_min = int(list(tree.iter('xmin'))[0].text)
    x_max = int(list(tree.iter('xmax'))[0].text)
    y_min = int(list(tree.iter('ymin'))[0].text)
    y_max = int(list(tree.iter('ymax'))[0].text)
    race = list(tree.iter('name'))[0].text.lower()

    return (x_min, y_min, x_max, y_max), race


def extract_cropped_images():
    os.makedirs(CROPPED_IMAGES_PATH, exist_ok=True)

    for image_path in glob.glob(f'{IMAGES_PATH}/*/*.jpg'):
        dimensions, race = extract_information_from_annotations(image_path)

        original_image = Image.open(image_path)
        cropped_image = original_image.crop(dimensions)

        cropped_image_path = f"{CROPPED_IMAGES_PATH}/{race}/" + image_path.split("/")[-1]
        os.makedirs(cropped_image_path.replace(image_path.split("/")[-1], ""), exist_ok=True)

        try:
            cropped_image.save(cropped_image_path)
        except OSError as e:
            if e.args == ('cannot write mode RGBA as JPEG',):
                print(f"Converting {cropped_image_path} to RGB.")
                cropped_image = cropped_image.convert('RGB')
                cropped_image.save(cropped_image_path)

    print(f"All cropped images have been extracted and saved under:{CROPPED_IMAGES_PATH}.\n")


def load_image(row):
    return cv2.imread(row['image_path'], 1)


def load_images():
    images_df = DataFrame()

    all_images = list(glob.glob(f"{CROPPED_IMAGES_PATH}/*/*.jpg"))
    images_df["image_path"] = all_images
    images_df["image"] = images_df.apply(load_image, axis=1)

    images_df["label_name"] = images_df["image_path"].apply(lambda path: path.split("/")[-2].lower())

    labels = [f.path.split("/")[-1].lower() for f in os.scandir(CROPPED_IMAGES_PATH) if f.is_dir()]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    images_df["label"] = label_encoder.transform(images_df["label_name"])

    return images_df


def print_images_dimensions(images_df):
    dimensions = []

    for image_path in images_df[['image_path']].values:
        image = Image.open(image_path[0])
        width, height = image.size

        dimensions.append({"width": width, "height": height})

    dimensions_df = DataFrame(dimensions)
    print(dimensions_df.describe())
    print("\n")


def resize_image(row):
    return cv2.resize(row['image'], (300, 300))


def convert_image_to_grayscale(row):
    return cv2.cvtColor(row['resized_image'], cv2.COLOR_BGR2GRAY)


def denoise_image(row):
    return cv2.fastNlMeansDenoising(row['grayscaled_image'], None, 10, 7, 21)


def equalize_histogram(row):
    return cv2.equalizeHist(row['grayscaled_image'])


def display_images_count_per_label(images_df):
    counts = []
    for label in images_df['label_name'].unique():
        counts.append({"label": label, "count": len(images_df[images_df['label_name'] == label])})

    counts_df = DataFrame(counts).sort_values("label")
    mean_count = sum(count['count'] for count in counts) / len(counts)

    fig = px.bar(counts_df, x='label', y='count', labels={'label': 'Race', 'count': 'Count'},
                 hover_data={'label': True, 'count': True})

    # counts_plot = (counts_df.plot(kind="bar", x="label", figsize=(15, 8), rot=0,
    #                               title=f"Count of images per label"))

    fig.add_hline(y=mean_count, line_dash="dash", line_color="green", annotation_text=f"Mean: {mean_count:.2f}")

    fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=False)

    os.makedirs(PLOTLY_PATH, exist_ok=True)
    fig.write_json(f"{PLOTLY_PATH}/images_count_per_race.json")



def plot_image_dimensions_distribution(images_df):
    dimensions = []

    for image_path in images_df[['image_path']].values:
        image = Image.open(image_path[0])
        width, height = image.size
        dimensions.append({"width": width, "height": height})

    dimensions_df = DataFrame(dimensions)
    dimensions_df['index'] = dimensions_df.index

    fig = px.line(dimensions_df, x='index', y=['width', 'height'],
                  labels={'index': 'Image Index', 'value': 'Dimension Value'})

    mean_width = dimensions_df["width"].mean()
    mean_height = dimensions_df["height"].mean()

    fig.add_hline(y=mean_width, line_dash="dash", line_color="blue", annotation_text=f"Mean Width: {mean_width:.2f}")
    fig.add_hline(y=mean_height, line_dash="dash", line_color="orange",
                  annotation_text=f"Mean Height: {mean_height:.2f}")

    fig.update_layout(xaxis_title=None, yaxis_title=None, showlegend=True)

    # Save the plot as an interactive HTML file
    os.makedirs(PLOTLY_PATH, exist_ok=True)
    fig.write_json(f"{PLOTLY_PATH}/images_dimensions.json")


if __name__ == '__main__':
    print("Starting analysis and preprocessing script.\n")

    # if not exists(CROPPED_IMAGES_PATH):
    #     extract_cropped_images()

    images_df = load_images()
    print(f"{len(images_df)} images have been loaded with {len(images_df['label_name'].unique())} different labels.\n")

    print("Displaying now the count of images per label.\n")
    # display_images_count_per_label(images_df)

    print("Displaying now the dimensions of the images.\n")
    # print_images_dimensions(images_df)

    # plot_image_dimensions_distribution(images_df)

    images_df = images_df.head(10)

    # print("Creating now resized images.\n")
    # # Resizing image to fit the 224x224 input size of most models
    images_df["resized_image"] = images_df.apply(resize_image, axis=1)
    #
    # print("Creating now grayscaled images.\n")
    # # Conversion of the image to grayscale as color is not a relevant information in race detection
    # images_df["grayscaled_image"] = images_df.apply(convert_image_to_grayscale, axis=1)
    #
    # print("Creating now denoised images.\n")
    # # Denoise image to improve its quality and reduce the impact of noise on the model
    # images_df["denoised_image"] = images_df.apply(denoise_image, axis=1)
    #
    # print("Creating now equalized images.\n")
    # # Equalize the histogram of the image to improve the contrast and make the features more visible
    # images_df["equalized_image"] = images_df.apply(equalize_histogram, axis=1)
    #
    for image_index in range(1, len(images_df) + 1):
        print(f"Displaying different states of image number:{image_index}.\n")
        image = images_df.iloc[image_index]

        # Extract the filename from the image path
        filename = os.path.basename(image['image_path'])

        # Construct the path to the original image in the resources/Images folder
        search_path = os.path.join(IMAGES_PATH, '**', filename)
        original_image_paths = glob.glob(search_path, recursive=True)

        if original_image_paths:
            original_image_path = original_image_paths[0]
        else:
            print(f"Image {filename} not found in {IMAGES_PATH}.")
            continue

        # Open the original image
        original_image = Image.open(original_image_path)

        resized_image_rgb = cv2.cvtColor(image['resized_image'], cv2.COLOR_BGR2RGB)

        f, ax = plt.subplots(1, 2, figsize=(9, 6))
        # f, ax = plt.subplots(2, 3, figsize=(9, 6))
        # ax[0].imshow(image['image'])
        ax[0].imshow(original_image)
        ax[1].imshow(resized_image_rgb)
        # ax[0, 2].axis('off')
        # ax[1, 0].imshow(image['grayscaled_image'], cmap='gray', vmin=0, vmax=255)
        # ax[1, 1].imshow(image['denoised_image'], cmap='gray', vmin=0, vmax=255)
        # ax[1, 2].imshow(image['equalized_image'], cmap='gray', vmin=0, vmax=255)

        plt.show()
        plt.close()

    print("Analysis and preprocessing script finished.\n")
