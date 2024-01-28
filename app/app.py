import streamlit as st
from PIL import Image
import app_function 
import time
import os
import matplotlib.pyplot as plt 

image_path = os.path.join(os.path.dirname(__file__), 'analytica.png')
st.sidebar.image(image_path, use_column_width=True)
st.sidebar.title("boneX")
st.sidebar.subheader("The newest Image Analytica product!")
#st.sidebar.text("boneX is a x-ray fracture classifier."
#       "In this first app interaction, boneX classifies upper limbs fractures for its location. In the next releases, it will also find and mark the exact fracture location."
#           "Upload an X-ray image, and try it now!")

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

        'Analysing your image...'
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.1)  
            progress_bar.progress(percent_complete + 1)
        'Done!'

        
        predictions = app_function.xray_image(resized_image)

        st.write("Predictions:")
        #st.bar_chart(predictions)
        fig, ax = plt.subplots()
        bars = ax.bar(predictions.keys(), predictions.values(), color='rosybrown')  
        ax.set_facecolor('none') 
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

