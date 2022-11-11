import streamlit as st
import yaml
import io
from predict_trial import load_model, get_prediction
from PIL import Image
from confirm_button_hack import cache_on_button_press

st.set_page_config(page_title="Upload_picture", page_icon="🎞️")

st.markdown("# Mask Classification Model")
st.markdown("### Upload picture you want to analyze")
st.sidebar.header("Upload_picture")

    
with open("config.yaml") as f:
    config = yaml.load(f, Loader = yaml.FullLoader)
model_mask, model_age, model_gender = load_model()
model_mask.eval()
model_age.eval()
model_gender.eval()
    
# TODO: File Uploader
uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])
uploaded_file
    
# TODO: 이미지 View
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image")
    with st.spinner("Classifying..."):
        _, y_mask = get_prediction(model_mask, image_bytes)
        _, y_age = get_prediction(model_age, image_bytes)
        _, y_gender = get_prediction(model_gender, image_bytes)
    y = 6 * y_mask + y_age + 3*y_gender
    label = config['classes'][y.item()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mask", label[0])
    col2.metric("Gender", label[1])
    if label[2] == "between 30 and 60":
        label[2] = "30 ~ 60"
    col3.metric("Age", label[2])   