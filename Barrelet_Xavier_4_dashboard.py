import io

import keras
import numpy as np
import streamlit as st
from PIL import Image
from keras import Model

MODEL_PATH = "model_vit.keras"

CACHED_RACES = {41: 'english_springer', 17: 'border_terrier', 33: 'collie', 52: 'great_pyrenees', 111: 'toy_terrier',
                88: 'pug', 104: 'staffordshire_bullterrier', 2: 'african_hunting_dog', 93: 'saluki',
                45: 'french_bulldog', 11: 'bernese_mountain_dog', 27: 'cardigan', 91: 'rottweiler', 114: 'weimaraner',
                82: 'old_english_sheepdog', 97: 'scottish_deerhound', 80: 'norwegian_elkhound',
                12: 'black-and-tan_coonhound', 62: 'keeshond', 87: 'pomeranian', 71: 'malamute', 73: 'maltese_dog',
                90: 'rhodesian_ridgeback', 44: 'flat-coated_retriever', 3: 'airedale', 118: 'wire-haired_fox_terrier',
                0: 'affenpinscher', 30: 'chow', 72: 'malinois', 115: 'welsh_springer_spaniel',
                20: 'bouvier_des_flandres', 7: 'basenji', 4: 'american_staffordshire_terrier',
                34: 'curly-coated_retriever', 94: 'samoyed', 116: 'west_highland_white_terrier', 112: 'vizsla',
                14: 'bloodhound', 39: 'english_foxhound', 51: 'great_dane', 37: 'dingo', 66: 'kuvasz', 65: 'komondor',
                98: 'sealyham_terrier', 46: 'german_shepherd', 106: 'standard_schnauzer', 109: 'tibetan_terrier',
                43: 'eskimo_dog', 28: 'chesapeake_bay_retriever', 36: 'dhole', 108: 'tibetan_mastiff',
                48: 'giant_schnauzer', 22: 'brabancon_griffon', 9: 'beagle', 42: 'entlebucher', 78: 'newfoundland',
                113: 'walker_hound', 76: 'miniature_poodle', 19: 'boston_bull', 15: 'bluetick', 56: 'irish_setter',
                101: 'siberian_husky', 57: 'irish_terrier', 38: 'doberman', 63: 'kelpie', 86: 'pembroke', 8: 'basset',
                92: 'saint_bernard', 70: 'lhasa', 21: 'boxer', 75: 'miniature_pinscher', 10: 'bedlington_terrier',
                67: 'labrador_retriever', 68: 'lakeland_terrier', 85: 'pekinese', 99: 'shetland_sheepdog',
                77: 'miniature_schnauzer', 59: 'irish_wolfhound', 29: 'chihuahua', 35: 'dandie_dinmont',
                60: 'italian_greyhound', 79: 'norfolk_terrier', 81: 'norwich_terrier', 89: 'redbone', 110: 'toy_poodle',
                18: 'borzoi', 55: 'ibizan_hound', 64: 'kerry_blue_terrier', 23: 'briard', 49: 'golden_retriever',
                6: 'australian_terrier', 69: 'leonberg', 103: 'soft-coated_wheaten_terrier', 13: 'blenheim_spaniel',
                58: 'irish_water_spaniel', 100: 'shih-tzu', 31: 'clumber', 40: 'english_setter', 5: 'appenzeller',
                32: 'cocker_spaniel', 1: 'afghan_hound', 95: 'schipperke', 84: 'papillon', 105: 'standard_poodle',
                54: 'groenendael', 119: 'yorkshire_terrier', 74: 'mexican_hairless', 50: 'gordon_setter', 26: 'cairn',
                102: 'silky_terrier', 83: 'otterhound', 107: 'sussex_spaniel', 53: 'greater_swiss_mountain_dog',
                25: 'bull_mastiff', 16: 'border_collie', 117: 'whippet', 96: 'scotch_terrier', 24: 'brittany_spaniel',
                47: 'german_short-haired_pointer', 61: 'japanese_spaniel'}

# CLOUD URL: https://project-7-dashboard.streamlit.app/

st.set_page_config(layout='wide', page_title="Dashboard de prédictions")
st.title("Dashboard de prédictions")

st.write("Ce dashboard permet de tester la prédiction de la race de chien de l'image téléversée.")

with st.container():
    with st.expander("Exploration du jeu de données"):
        st.write("Voici le nombre d'images par race de chien avec leur moyenne:")
        st.image("images/images_count_per_race.png", caption="Nombre d'images par race de chien")

        st.write("Voici les dimensions des images:")
        st.image("images/images_dimensions.png", caption="Dimensions des images")

        st.write('Voici quelques images du set de données originale puis après recadrement et redimensionnement.')
        st.image("images/example_1.png", caption="Premier exemple")
        st.image("images/example_2.png", caption="Deuxième exemple")
        st.image("images/example_3.png", caption="Troisième exemple")

model: Model = keras.models.load_model(MODEL_PATH, compile=False)

img_file_buffer = st.file_uploader("Téléversez une image du set de données", type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()

    img = Image.open(io.BytesIO(bytes_data))
    img = img.convert('RGB')
    img = img.resize((224, 224))
    input_arr = keras.utils.img_to_array(img)
    image_batch = np.array([input_arr])

    predictions = model.predict(image_batch)

    preds_list = list(predictions[0])
    predicted_race_id = preds_list.index(max(preds_list))

    st.text(f"Race prédite: {CACHED_RACES[predicted_race_id]}.\n")
