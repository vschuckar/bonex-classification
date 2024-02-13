import streamlit as st
from PIL import Image
import app_function
import time
import os
import altair as alt
import pandas as pd

model_url = 'https://raw.githubusercontent.com/vschuckar/bonex-classification/main/models/best_model_-10.h5'
model = app_function.load_cached_model(model_url)

image_path = os.path.join(os.path.dirname(__file__), 'analytica.png')
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.title("boneX")
st.sidebar.subheader("The newest Image Analytica product!")

multiline_text = """
### Overview

boneX is a state-of-the-art X-ray fracture classifier. 

In this app, you can upload an X-ray image of an upper limb, and boneX will analyze it to classify fractures in different locations.

### How to Use

1. Upload an X-ray image using the file uploader.
2. Click the 'Make predictions!' button to analyze the image.
3. boneX will provide predictions for various fracture locations.

Enjoy using boneX for quick and accurate fracture classification!
"""

st.markdown(multiline_text, unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an X-ray image.", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    resized_image = image.resize((299, 299))
    st.image(resized_image, caption="Uploaded X-ray image.", use_column_width=True)

    if st.button("Make predictions!"):
        with st.spinner("Making predictions..."):
            predictions = app_function.xray_image(resized_image, model)

        st.success("Done!")

        st.write("Predictions:")
        df = pd.DataFrame(list(predictions.items()), columns=['Category', 'Probability'])
        chart = alt.Chart(df).mark_bar(color='#6F2727').encode(
            x=alt.X('Category:N', title=None),
            y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1])),
            tooltip=['Category', 'Probability']
        ).properties(
            width=alt.Step(100),
            height=450
        ).configure_axis(
            labelAngle=45,
        )

        st.altair_chart(chart)

