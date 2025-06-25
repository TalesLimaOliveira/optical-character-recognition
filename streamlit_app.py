import streamlit as st
import torch
import numpy as np
from src.nn_draw import draw_network_visualization
from src.utils import load_model, load_testset, pick_new_image

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
st.set_page_config(layout="wide",page_title="Optical Character Recognition",)
left_col, center_col, right_col = st.columns([1, 3, 1])

# --- Top menu navigation ---
with left_col:
    st.markdown('''
    <div style="text-align:left; margin-bottom:2em; margin-top:0.5em;">
      <span style="color:#888; font-size:1.08em; font-weight:bold;">See other projects:</span><br>
      <span style="color:#444; background:#e5e5e5; border-radius:6px; font-size:1.1em; display:inline-block; margin-bottom:0.3em; margin-left:0.7em; padding:2px 8px; width:auto;">• Character Recognition</span>
      <a href="https://digit-gen-taleslimaoliveira.streamlit.app/" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-bottom:0.3em; margin-left:0.7em;">• Digit Generator</a>
      <a href="https://github.com/TalesLimaOliveira" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-left:0.7em;">• My Github</a>
    </div>
    ''', unsafe_allow_html=True)
    
# <a href="https://gen-ocr-taleslimaoliveira.streamlit.app/" style="color:#1a73e8; text-decoration:none; font-size:1.1em; display:block; margin-left:0.7em;">• Gen and OCR</a>   
    
# --- Title and description ---
with center_col:
    st.title('Optical Character Recognition')
    st.markdown('<div style="font-size:16px; margin-bottom:0px; margin-top:-16px; text-align:left;">by <a href="https://github.com/TalesLimaOliveira" target="_blank">Tales Lima Oliveira</a></div>', unsafe_allow_html=True)
    st.write("Predict what character is using MNIST dataset.")
    st.markdown('<div style="margin-bottom:0.5em; margin-top:1.5em;"></div>', unsafe_allow_html=True)


    # Center column for image and prediction
    data_cols = st.columns([2,1,2,2], gap="large")
    with data_cols[1]:
        img_to_show = img.squeeze().numpy() * 0.3081 + 0.1307
        st.image(img_to_show, width=128, caption=f'True label: {label}')
    with data_cols[2]:
        st.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
        if st.button("Random Image", key="generate_btn", use_container_width=True):
            pick_new_image(testset)
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
    st.markdown('<h3 style="text-align:left; margin-bottom:0.5em; margin-top:0em;">Neural Network Visualization:</h3>', unsafe_allow_html=True)
    draw_network_visualization(model, img)