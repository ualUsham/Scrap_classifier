import streamlit as st
from PIL import Image
import numpy as np
import keras
model=keras.models.load_model("scrap_classifier.keras")

st.title("Scrap Classifier")
st.write("**Upload an image of Aluminium, Copper or Steel Scrap.**")

# Upload image
image_upload = st.file_uploader("image", type=["jpg", "jpeg", "png"],label_visibility='hidden')

if image_upload is not None:
    # Open the image using PIL
    image = keras.preprocessing.image.load_img(image_upload,target_size=(200,200))
    # Display the uploaded image
    st.image(image, caption="**Uploaded Scrap Image**", width=250)
    #preprocess the image
    ar=keras.preprocessing.image.array_to_img(image)
    ar=np.expand_dims(ar,axis=0)
    #predict
    val=model.predict(ar)
    st.write('**Answer:**')
    if np.argmax(val)==1:
        st.write('**Its aluminium scrap**')
    if np.argmax(val)==0:
        st.write('**Its copper scrap**')
    if np.argmax(val)==2:
        st.write("**Its steel scrap**")
    



