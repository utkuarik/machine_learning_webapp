from concurrent.futures import process
import streamlit as st
import pandas as pd
import os
import collections.abc as container_abcs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import sys
from pandas.errors import ParserError
import time
import altair as altpi
import matplotlib.cm as cm
import graphviz
import base64
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.layouts import layout
from bokeh.plotting import figure
from bokeh.models import Toggle, BoxAnnotation
from bokeh.models import Panel, Tabs
from bokeh.palettes import Set3

# Keras specific
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import torch
import clip
from PIL import Image
import requests
import io

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import models
from diffusers import StableDiffusionPipeline
from torch.cuda.amp import autocast

from rembg import remove
import base64

from image_utils import *
# from faceeditor.face_model import *
# from faceeditor.face_test import *
# from faceeditor.makeup import *


import os
import sys
sys.path.insert(0, './faceeditor/')
sys.path.append(os.getcwd())

def load_chatbot_model():
    st.title('ChatBot')

    @st.cache_resource()
    def load_model():
        with st.spinner("Model is loading"):
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium")
            st.balloons()
        return tokenizer, model

    tokenizer, model = load_model()

    chat_history_ids = None
    prompt = st.text_input("please provide a prompt")

    if 'count' not in st.session_state or st.session_state.count == 6:
        st.session_state.count = 0

        st.session_state.chat_history_ids = None
        st.session_state.old_response = ''
    else:
        st.session_state.count += 1

    st.write(st.session_state.count)
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        prompt + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids],
                              dim=-1) if st.session_state.count > 1 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    st.session_state.chat_history_ids = model.generate(
        bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    # pretty print last ouput tokens from bot
    st.write("Bot: {}".format(response))


@st.cache_resource()
def load_image_model():
    with st.spinner("Model is loading"):
        model, preprocess, device = load_clip_model()
        st.balloons()

    return model, preprocess, device

    return model, process, device


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    return model, preprocess, device


class PreProcessor:

    def __init__(self) -> None:
        self.data = None
        self.selection = None

    def prepare_data_tabular(self, controller, split_data, train_test):
        # Reduce data size
        data = controller.data[controller.features]
        data = data.sample(frac=round(split_data/100, 2))

        # Impute nans with mean for numeris and most frequent for categoricals
        cat_imp = SimpleImputer(strategy="most_frequent")
        if len(data.loc[:, data.dtypes == 'object'].columns) != 0:
            data.loc[:, data.dtypes == 'object'] = cat_imp.fit_transform(
                data.loc[:, data.dtypes == 'object'])
        imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        data.loc[:, data.dtypes != 'object'] = imp.fit_transform(
            data.loc[:, data.dtypes != 'object'])

        # One hot encoding for categorical variables
        cats = data.dtypes == 'object'
        le = LabelEncoder()
        for x in data.columns[cats]:
            sum(pd.isna(data[x]))
            data.loc[:, x] = le.fit_transform(data[x])
        onehotencoder = OneHotEncoder()
        data.loc[:, ~cats].join(pd.DataFrame(data=onehotencoder.fit_transform(
            data.loc[:, cats]).toarray(), columns=onehotencoder.get_feature_names_out()))

        # Set target column
        target_options = data.columns
        self.chosen_target = st.sidebar.selectbox(
            "Please choose target column", (target_options))

        # Standardize the feature data
        X = data.loc[:, data.columns != self.chosen_target]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)
        X = pd.DataFrame(scaler.transform(X))
        X.columns = data.loc[:, data.columns != self.chosen_target].columns
        y = data[self.chosen_target]

        # Train test split
        try:
            controller.X_train, controller.X_test, controller.y_train, controller.y_test = train_test_split(
                X, y, test_size=(1 - train_test/100), random_state=42)
        except:
            st.markdown('<span style="color:red">With this amount of data and split size the train data will have no records, <br /> Please change reduce and split parameter <br /> </span>', unsafe_allow_html=True)

# Main Predicor class


class Predictor:

    def __init__(self) -> None:
        self.data = None
        self.selection = None

    def clip_predictor(self, image, labels):
        model, preprocess, device = load_image_model()
        image = preprocess(image).unsqueeze(0).to(device)
        text = clip.tokenize(labels).to(device)

        with torch.no_grad():

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # st.write("Label probs:", probs)
        return probs

    # Classifier type and algorithm selection

    def set_classifier_properties(self):
        self.type = st.sidebar.selectbox(
            "Algorithm type", ("Classification", "Regression", "Clustering"))
        if self.type == "Regression":
            self.chosen_classifier = st.sidebar.selectbox(
                "Please choose a classifier", ('Random Forest', 'Linear Regression', 'Neural Network'))
            if self.chosen_classifier == 'Random Forest':
                self.n_trees = st.sidebar.slider('number of trees', 1, 1000, 1)
            elif self.chosen_classifier == 'Neural Network':
                self.epochs = st.sidebar.slider('number of epochs', 1, 100, 10)
                self.learning_rate = float(
                    st.sidebar.text_input('learning rate:', '0.001'))
        elif self.type == "Classification":
            self.chosen_classifier = st.sidebar.selectbox(
                "Please choose a classifier", ('Logistic Regression', 'Naive Bayes', 'Neural Network'))
            if self.chosen_classifier == 'Logistic Regression':
                self.max_iter = st.sidebar.slider('max iterations', 1, 100, 10)
            elif self.chosen_classifier == 'Neural Network':
                self.epochs = st.sidebar.slider('number of epochs', 1, 100, 10)
                self.learning_rate = float(
                    st.sidebar.text_input('learning rate:', '0.001'))
                self.number_of_classes = int(
                    st.sidebar.text_input('Number of classes', '2'))

        elif self.type == "Clustering":
            pass

    # Model training and predicitons
    def predict(self, predict_btn):

        if self.type == "Regression":
            if self.chosen_classifier == 'Random Forest':
                self.alg = RandomForestRegressor(
                    max_depth=2, random_state=0, n_estimators=self.n_trees)
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == 'Linear Regression':
                self.alg = LinearRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == 'Neural Network':
                model = Sequential()
                model.add(Dense(500, input_dim=len(
                    self.X_train.columns), activation='relu',))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(1))

                # optimizer = keras.optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(loss="mean_squared_error",
                              optimizer='adam', metrics=["mean_squared_error"])
                self.model = model.fit(
                    self.X_train, self.y_train, epochs=self.epochs, batch_size=40)
                self.predictions = model.predict(self.X_test)
                self.predictions_train = model.predict(self.X_train)

        elif self.type == "Classification":
            if self.chosen_classifier == 'Logistic Regression':
                self.alg = LogisticRegression()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == 'Naive Bayes':
                self.alg = GaussianNB()
                self.model = self.alg.fit(self.X_train, self.y_train)
                predictions = self.alg.predict(self.X_test)
                self.predictions_train = self.alg.predict(self.X_train)
                self.predictions = predictions

            elif self.chosen_classifier == 'Neural Network':
                model = Sequential()
                model.add(Dense(500, input_dim=len(
                    self.X_train.columns), activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(50, activation='relu'))
                model.add(Dense(self.number_of_classes, activation='softmax'))

                optimizer = tf.keras.optimizers.SGD(
                    lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(
                    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model = model.fit(
                    self.X_train, self.y_train, epochs=self.epochs, batch_size=40)

                self.predictions = model.predict_classes(self.X_test)
                self.predictions_train = model.predict_classes(self.X_train)

        result = pd.DataFrame(
            columns=['Actual', 'Actual_Train', 'Prediction', 'Prediction_Train'])
        result_train = pd.DataFrame(
            columns=['Actual_Train', 'Prediction_Train'])
        result['Actual'] = self.y_test
        result_train['Actual_Train'] = self.y_train
        result['Prediction'] = self.predictions
        result_train['Prediction_Train'] = self.predictions_train
        result.sort_index()
        self.result = result
        self.result_train = result_train

        return self.predictions, self.predictions_train, self.result, self.result_train

    # Get the result metrics of the model
    def get_metrics(self):
        self.error_metrics = {}
        if self.type == 'Regression':
            self.error_metrics['MSE_test'] = mean_squared_error(
                self.y_test, self.predictions)
            self.error_metrics['MSE_train'] = mean_squared_error(
                self.y_train, self.predictions_train)
            return st.markdown('### MSE Train: ' + str(round(self.error_metrics['MSE_train'], 3)) +
                               ' -- MSE Test: ' + str(round(self.error_metrics['MSE_test'], 3)))

        elif self.type == 'Classification':
            self.error_metrics['Accuracy_test'] = accuracy_score(
                self.y_test, self.predictions)
            self.error_metrics['Accuracy_train'] = accuracy_score(
                self.y_train, self.predictions_train)
            return st.markdown('### Accuracy Train: ' + str(round(self.error_metrics['Accuracy_train'], 3)) +
                               ' -- Accuracy Test: ' + str(round(self.error_metrics['Accuracy_test'], 3)))

    # Plot the predicted values and real values
    def plot_result(self):

        output_file("slider.html")

        s1 = figure(width=800, height=500, background_fill_color="#fafafa")
        s1.circle(self.result_train.index, self.result_train.Actual_Train,
                  size=12, color="Black", alpha=1, legend_label="Actual")
        s1.triangle(self.result_train.index, self.result_train.Prediction_Train,
                    size=12, color="Red", alpha=1, legend_label="Prediction")
        tab1 = Panel(child=s1, title="Train Data")

        if self.result.Actual is not None:
            s2 = figure(width=800, height=500, background_fill_color="#fafafa")
            s2.circle(self.result.index, self.result.Actual, size=12,
                      color=Set3[5][3], alpha=1, legend_label="Actual")
            s2.triangle(self.result.index, self.result.Prediction, size=12,
                        color=Set3[5][4], alpha=1, legend_label="Prediction")
            tab2 = Panel(child=s2, title="Test Data")
            tabs = Tabs(tabs=[tab1, tab2])
        else:

            tabs = Tabs(tabs=[tab1])

        st.bokeh_chart(tabs)

    # File selector module for web app

    def file_selector(self):
        file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if file is not None:
            data = pd.read_csv(file)
            return data
        else:
            st.text("Please upload a csv file")

    def method_selector(self):
        selection = st.sidebar.selectbox(
            'How would you like to be contacted?',
            ('Image Recognition', 'Image Generation', 'Image Editor', 'Tabular Data Prediction', 'ChatBot'))

        return selection

    def print_table(self):
        if len(self.result) > 0:
            result = self.result[['Actual', 'Prediction']]
            st.dataframe(result.sort_values(
                by='Actual', ascending=False).style.highlight_max(axis=0))

    def set_features(self):
        self.features = st.multiselect(
            'Please choose the features including target variable that go into the model', self.data.columns)


if __name__ == '__main__':

    # Define controller objects
    controller = Predictor()
    preprocessor = PreProcessor()

    try:
        # controller.data = controller.file_selector()
        controller.selection = controller.method_selector()

        if controller.selection == "Tabular Data Prediction":

            st.title('Tabular Data Prediction')

            controller.data = controller.file_selector()
            if controller.data is not None:  # Check if user provided a file, then proceed
                split_data = st.sidebar.slider(
                    'Randomly reduce data size %', 1, 100, 10)
                train_test = st.sidebar.slider('Train-test split %', 1, 99, 66)
                controller.set_features()
                st.write(controller.features)
                if len(controller.features) > 1:

                    preprocessor.prepare_data_tabular(
                        controller, split_data, train_test)
                    controller.set_classifier_properties()
                    predict_btn = st.sidebar.button('Predict')

        elif controller.selection == "Image Recognition":

            st.title('Image Recognition')
            url = st.text_input("Please paste image url")

            if len(url) > 0:  # Check if user provided a url, then proceed
                resp = requests.get(url)
                image = Image.open(io.BytesIO(requests.get(url).content))
                st.image(image, width=300, caption='Provided Image')
                labels = st.text_input("Enter classes to predict with commas")

                if len(labels) > 0:  # Check if user provided classes, then proceed
                    labels = labels.replace(" ", "").split(",")
                    probs = controller.clip_predictor(image, labels)

                    st.subheader(':blue[Results:]')
                    st.markdown("Class probabilities are:",
                                unsafe_allow_html=False)
                    st.write(dict(zip(labels, probs[0])))

        # elif controller.selection == "Image Generation":

            # import jax
            # import numpy as np
            # from flax.jax_utils import replicate
            # from flax.training.common_utils import shard
            # import PIL
            # import requests
            # from io import BytesIO

            # from diffusers import FlaxStableDiffusionInpaintPipeline

            # def download_image(url):
            #     response = requests.get(url)
            #     return PIL.Image.open(BytesIO(response.content)).convert("RGB")

            # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
            # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

            # init_image = download_image(img_url).resize((512, 512))
            # mask_image = download_image(mask_url).resize((512, 512))

            # pipeline, params = FlaxStableDiffusionInpaintPipeline.from_pretrained("xvjiarui/stable-diffusion-2-inpainting")

            # prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
            # prng_seed = jax.random.PRNGKey(0)
            # num_inference_steps = 50

            # num_samples = jax.device_count()
            # prompt = num_samples * [prompt]
            # init_image = num_samples * [init_image]
            # mask_image = num_samples * [mask_image]
            # prompt_ids, processed_masked_images, processed_masks = pipeline.prepare_inputs(prompt, init_image, mask_image)

            # # shard inputs and rng
            # params = replicate(params)
            # prng_seed = jax.random.split(prng_seed, jax.device_count())
            # prompt_ids = shard(prompt_ids)
            # processed_masked_images = shard(processed_masked_images)
            # processed_masks = shard(processed_masks)

            # images = pipeline(prompt_ids, processed_masks, processed_masked_images, params, prng_seed, num_inference_steps, jit=True).images
            # images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))

            # st.image(images, width = 300, caption='Provided Image', clamp=True, channels='RBG')

        elif controller.selection == "ChatBot":

            load_chatbot_model()

        elif controller.selection == "Image Editor":

            st.header("Image Editor")
            uploaded_image = st.sidebar.file_uploader(
                "Upload an image", type=["png", "jpg", "jpeg"])

            if uploaded_image:
                image = Image.open(uploaded_image)
                original_image = image.copy()

                col11, col12, col13 = st.columns(3)
                with col12:
                    st.markdown('**Uploaded Image**:')
                    st.image(image)
                
                color = st.sidebar.color_picker('Choose a background color', '#DEC0B3')
                rvbg = st.sidebar.button('Remove Background')

                manipulation_list = st.multiselect("""Choose manipulations
                            """ ,
                ['Grayscale', 'Blur', 'Brightness', 'Contrast', 'Vignette', 'Saturation', 'Sepia'],
                None)

                if "Grayscale" in manipulation_list:
                    image = grayscale(image)

                if "Blur" in manipulation_list:
                    image = blur(image)

                if "Brightness" in manipulation_list:
                    level = st.slider("Brightness level",
                                      min_value=0.1, max_value=3.0, step=0.1)
                    image = brightness(image, level)

                if "Contrast" in manipulation_list:
                    level = st.slider("Contrast level",
                                      min_value=0.1, max_value=3.0, step=0.1)
                    image = contrast(image, level)

                if "Saturation" in manipulation_list:
                    level = st.slider("Contrast level",
                                      min_value=0.1, max_value=3.0, step=0.1)
                    image = saturation(image, level)

                if "Sepia" in manipulation_list:
                    image = sepia(image)

                if "Crop" in manipulation_list:
                    width = st.number_input("Enter the width", value=200)
                    height = st.number_input("Enter the height", value=200)
                    image = crop(image, width, height)

                if "Vignette" in manipulation_list:
                    image = add_vignette(image)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_image)

                with col2:
                    st.image(image)

                if rvbg:
                    size = image.size
                    edited_image = Image.new("RGB", size, color)
                    out = remove(image)
                    edited_image.paste(out, mask=out)
                    st.image(edited_image)

                edited_image = image
                img_byte_arr = io.BytesIO()
                edited_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                btn = st.download_button(
                    label="Download image",
                    data=img_byte_arr,
                    file_name="downloaded_image.png",
                    mime="image/png"
                )

    except (AttributeError, ParserError, KeyError) as e:
        st.write(e)
        st.markdown('<span style="color:blue">WRONG FILE TYPE</span>',
                    unsafe_allow_html=True)

    if controller.data is not None and len(controller.features) > 1:
        if predict_btn:
            st.sidebar.text("Progress:")
            my_bar = st.sidebar.progress(0)
            predictions, predictions_train, result, result_train = controller.predict(
                predict_btn)
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)

            controller.get_metrics()
            controller.plot_result()
            controller.print_table()

            data = controller.result.to_csv(index=False)
            # b64 = base64.b64encode(data.encode()).decode()  # some strings <-> bytes conversions necessary here
            # href = f'<a href="data:file/csv;base64,{b64}">Download Results</a> (right-click and save as &lt;some_name&gt;.csv)'
            # st.sidebar.markdown(href, unsafe_allow_html=True)
            st.sidebar.download_button(
                "Download results as csv", data, file_name="results.csv")

    if controller.data is not None:
        if st.sidebar.checkbox('Show raw data'):
            st.subheader('Raw data')
            st.write(controller.data)
