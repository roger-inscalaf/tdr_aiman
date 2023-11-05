import streamlit as st
import tensorflow as tf
from PIL import Image
from img_classification import teachable_machine_classification

st.title("Diagnòstic MRI de tumors cerebrals amb IRM")
st.header("Brain Tumor MRI Classification")

uploaded_file = st.file_uploader("Puja una imatge IRM en format JPG per diagnosticar si conté alguna tipologia (o no) de càncer cerebral (Glioma, Meningioma o Pituitari)", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imatge IRM pujada', use_column_width=True)
    st.write("")
    st.write("Classificant...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 0:
        st.subheader("L'escàner IRM té un glioma")
    if label == 1:
        st.subheader("L'escàner IRM té un meningioma")
    if label == 2:
        st.subheader("L'escàner IRM té un tumor pituitàri")
    if label == 3:
        st.subheader("L'escàner és sa")
        
