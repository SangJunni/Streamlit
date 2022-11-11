import streamlit as st
import torch
from model_trial import MyEfficientNet, MyResNet_mask, MyResNet_age, MyResNet_gender
import yaml
from typing import Tuple
from utils_trial import transform_image

@st.cache
def load_model() -> MyEfficientNet:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_mask = MyResNet_mask(num_classes = 3).to(device)
    model_age = MyResNet_age(num_classes = 3).to(device)
    model_gender = MyResNet_gender(num_classes = 2).to(device)

    return model_mask, model_age, model_gender

def get_prediction(model, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = transform_image(image_bytes=image_bytes).to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat