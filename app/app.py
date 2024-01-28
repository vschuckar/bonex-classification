import streamlit as st
from PIL import Image
import app_function 
import time

st.sidebar.image("/Users/vschuckar/Desktop/data_analytics/Week_9/final_project/analytica.png", use_column_width=True)
st.title("boneX")
st.subheader("The newest Image Analytica product!")
st.text("boneX is a x-ray fracture classifier."
        "In this first app interaction, boneX classifies upper limbs fractures for its location. In the next releases, it will also find and mark the exact fracture location."
            "Upload an X-ray image, and try it now!")

uploaded_file = st.file_uploader("Choose an X-ray image.", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray image.", use_column_width=True)

    
    if st.button("Make predictions!"):

        'Analysing your image...'
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.03)  
            progress_bar.progress(percent_complete + 1)
        'Done!'

        
        predictions = app_function.xray_image(image)

        st.write("Predictions:")
        st.bar_chart(predictions)

