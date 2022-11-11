#pip install streamlit-camera-input-live
import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
import torch
import torchvision
import yaml
from predict_trial import get_prediction, load_model

##상위 디렉토리의 파일 가져오기
import sys
sys.path.append('AI_tech/Boostcamp-AI-Tech-Product-Serving/part2/02-speical_mission')
import predict_trial

## 각 클래스를 불러오기
with open("config.yaml") as f:
    config = yaml.load(f, Loader = yaml.FullLoader)
model_mask, model_age, model_gender = predict_trial.load_model()
model_mask.eval()
model_age.eval()
model_gender.eval()

"# Real-time classification Demo"
"### Try any people to watch your webcam"
## 실시간 캡처
image = camera_input_live()

if image is not None:
    st.image(image)
    bytes_data = image.getvalue()
    st.write("Classifying")
    
    _, y_mask = predict_trial.get_prediction(model_mask, bytes_data)
    _, y_age = predict_trial.get_prediction(model_age, bytes_data)
    _, y_gender = predict_trial.get_prediction(model_gender, bytes_data)
    y = 6 * y_mask + y_age + 3*y_gender
    label = config['classes'][y.item()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Mask", label[0])
    col2.metric("Gender", label[1])
    if label[2] == "between 30 and 60":
        label[2] = "30 ~ 60"
    col3.metric("Age", label[2])