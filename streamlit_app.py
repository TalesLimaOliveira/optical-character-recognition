import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
from src.network import Net
from src.nn_draw import draw_network_visualization

# --- Model and dataset loading utilities ---
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load('models/mnist_classifier_final.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_testset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# --- Utility to pick a new random image ---
def pick_new_image():
    st.session_state.img_idx = np.random.randint(0, len(testset))

# --- Initialization ---
model = load_model()
testset = load_testset()

if 'img_idx' not in st.session_state:
    st.session_state.img_idx = np.random.randint(0, len(testset))

# --- Get image and prediction ---
img, label = testset[st.session_state.img_idx]
img_input = img.unsqueeze(0)
with torch.no_grad():
    output, _, _ = model(img_input)
    probs = torch.softmax(output, dim=1).numpy().flatten()
    pred = int(np.argmax(probs))


# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Optical Character Recognition")
left_col, center_col, right_col = st.columns([1, 3, 1])

# --- Title and description ---
with center_col:
    st.title('Optical Character Recognition')
    st.markdown('<div style="font-size:16px; margin-bottom:0px; margin-top:-16px; text-align:left;">by <a href="https://github.com/TalesLimaOliveira" target="_blank">Tales Lima Oliveira</a></div>', unsafe_allow_html=True)
    st.write("Predict what character is using MNIST dataset.")


    # Center column for image and prediction
    data_cols = st.columns([2,1,2,2], gap="small")
    with data_cols[1]:
        img_to_show = img.squeeze().numpy() * 0.3081 + 0.1307
        st.image(img_to_show, width=128, caption=f'True label: {label}')
    with data_cols[2]:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        if st.button("Select another random image", key="generate_btn", use_container_width=True):
            pick_new_image()
        st.markdown("""
            <style>
            div[data-testid=\"stButton\"] button {
                background-color: #21ba45 !important;
                color: white !important;
                font-size: 1.2em !important;
                padding: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

        is_correct = pred == label
        if is_correct:
            st.markdown(f"<div style='text-align:center;'><span style='color:white;font-size:1.2em'>Prediction: </span> <b><span style='color:green;font-size:1.5em'>{pred} (Correct)</span></b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:center;'><span style='color:white;font-size:1.2em'>Prediction: </span> <b><span style='color:red;font-size:1.5em'>{pred} (Wrong)</span></b></div>", unsafe_allow_html=True)

# Neural network visualization
with center_col:
    st.markdown('<h2 style="text-align:left; margin-bottom:0.5em; margin-top:1.5em;">Neural Network Visualization:</h2>', unsafe_allow_html=True)
    draw_network_visualization(model, img)