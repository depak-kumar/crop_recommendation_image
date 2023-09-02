import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
# Load your dataset here (replace this with your actual dataset)
df = pd.read_csv("/content/drive/MyDrive/training3csv file.csv")

# Split the dataset into features and labels
X = df.drop('label', axis=1)
y = df['label']

# Perform one-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['ph', 'humidity'])

# Train a machine learning model (Random Forest in this example)
model = RandomForestClassifier()
model.fit(X, y)
# Define the Streamlit app
st.title("Crop Recommendation App")
st.image("https://th.bing.com/th/id/OIP.X0sEZ2T_IMX3EST1DaKQ-QHaEj?w=303&h=187&c=7&r=0&o=5&dpr=1.3&pid=1.7",width=300,caption='Crop Recommendation model')
st.header("Data File")
st.write(df)



st.sidebar.header("User Input")
# soil_type = st.sidebar.slider("Select Soil Type", df['ph'].unique())
soil_type = st.sidebar.slider("Select Soil Type",0,10,4,1)
climate = st.sidebar.selectbox("Select Climate", df['humidity'].unique())
rainfall = st.sidebar.slider("Enter Average Rainfall (mm)", min_value=10, max_value=300)
temperature = st.sidebar.slider("Enter Average Temperature (°C)", min_value=0, max_value=50)


# # Function to recommend crops
# def recommend_crop(soil_type, climate, rainfall, temperature):
#     # Prepare the input data
#     input_data = {
#         'Soil_Type': [soil_type],
#         'Climate': [climate],
#         'Rainfall': [rainfall],
#         'Temperature': [temperature]
#     }
#     input_df = pd.DataFrame(input_data)
    
#     # Convert categorical variables to numerical using one-hot encoding
#     input_df = pd.get_dummies(input_df, columns=['Soil_Type', 'Climate'])
    
#     # Make a prediction using the trained model
#     predicted_crop = model.predict(input_df)
    
#     return predicted_crop[0]

def recommend_crop(soil_type, climate, rainfall, temperature):
    # Prepare the input data
    input_data = {
        'Soil_Type': [soil_type],
        'Climate': [climate],
        'Rainfall': [rainfall],
        'Temperature': [temperature]
    }
    input_df = pd.DataFrame(input_data)

    # Ensure that input_df has the same columns as the training data
    # If necessary, add missing columns with default values (e.g., all zeros)
    train_columns = X.columns  # Get the columns used during training
    missing_columns = set(train_columns) - set(input_df.columns)

    for column in missing_columns:
        input_df[column] = 0

    # Reorder columns to match the order used during training
    input_df = input_df[train_columns]

    # Make a prediction using the trained model
    predicted_crop = model.predict(input_df)[0]

    # Predict class probabilities
    class_probabilities = model.predict_proba(input_df)[0]

    return predicted_crop, class_probabilities




# Display the recommendation
if st.sidebar.button("Recommend"):
    recommended_crop = recommend_crop(soil_type, climate, rainfall, temperature)
    st.success(f"We recommend planting: **{recommended_crop}**")

# Additional information
st.sidebar.markdown("### About")
st.sidebar.info(
    "This is a simple crop recommendation app. It provides crop recommendations based on the user's input for soil type, climate, rainfall, and temperature."
)
# Function to recommend crops











# Additional information
st.sidebar.markdown("### About")
st.sidebar.info(
    "This is a simple crop recommendation app. It provides crop recommendations based on the user's input for soil type, climate, rainfall, and temperature."
)

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Title for your Streamlit app
st.title("Crop Recommendation from Images")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Function to process and classify the image
    # @st.cache(allow_output_mutation=True)
    def classify_image(img):
        # Load a pre-trained deep learning model (e.g., a CNN)
        model = tf.keras.applications.ResNet50(weights='imagenet')
        
        # Preprocess the image
        img = Image.open(img)
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Make a prediction
        prediction = model.predict(img_array)
        class_name = tf.keras.applications.resnet50.decode_predictions(prediction)[0][0][1]
        
        return class_name

    if st.button("Classify"):
        class_name = classify_image(uploaded_image)
        st.success(f"The image is classified as: {class_name} crop")



# import streamlit as st
# from PIL import Image
# import tensorflow as tf
# import numpy as np

# # Load a pre-trained deep learning model
# model = tf.keras.applications.MobileNetV2(weights="imagenet")

# # Title for your Streamlit app
# st.title("Plant Recommendation from Images")

# # Upload an image
# uploaded_image1 = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_image1 is not None:
#     st.image(uploaded_image1, caption="Uploaded Image", use_column_width=True)

#     # Function to process and classify the image
#     # @st.cache(allow_output_mutation=True)
#     def classify_image(img):
#         # Preprocess the image
#         img = Image.open(img)
#         img = img.resize((224, 224))
#         img_array = np.array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
#         # Make a prediction
#         prediction = model.predict(img_array)
#         class_name = tf.keras.applications.mobilenet_v2.decode_predictions(prediction)[0][0][1]
        
#         return class_name

#     if st.button("Classify"):
#         class_name = classify_image(uploaded_image1)
#         st.success(f"The image is classified as: {class_name}")

st.header("Made by DEEPAK KUMAR")