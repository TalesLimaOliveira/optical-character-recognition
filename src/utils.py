import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np
from src.network import Net
import streamlit as st

def load_model():
    """Load the trained MNIST model (cached)."""
    @st.cache_resource
    def _load():
        model = Net()
        model.load_state_dict(torch.load('models/mnist_classifier_final.pth', map_location='cpu'))
        model.eval()
        return model
    return _load()

def load_testset():
    """Load the MNIST test set (cached)."""
    @st.cache_resource
    def _load():
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return _load()

def pick_new_image(testset):
    """Pick a new random image index for the session state."""
    st.session_state.img_idx = np.random.randint(0, len(testset))
