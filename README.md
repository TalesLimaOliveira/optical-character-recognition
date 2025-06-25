# optical-character-recognition
OCR - Optical Character Recognition

# MNIST Neural Network Visualizer

This project is a web application for Optical Character Recognition (OCR) using a neural network trained on the MNIST dataset. The app is built with Streamlit and allows users to visualize how a neural network processes handwritten digit images.

## Features
- Loads a pre-trained neural network model (`mnist_classifier_final.pth`) trained on MNIST.
- Displays a random image from the MNIST test set as input.
- Visualizes the activations of the hidden layers (perceptrons) for the selected image.
- Shows the output layer (class probabilities) and highlights the predicted digit.
- Allows users to select and display another random image with a button.

## Project Structure
```
├── streamlit_app.py           # Streamlit web app (UI/front-end)
├── src/
│   ├── network.py             # Neural network architecture definition
│   ├── nn_draw.py             # Neural network visualization logic
│   └── utils.py               # Model/data loading and utility functions
├── models/
│   └── mnist_classifier_final.pth # Pre-trained model weights (not included in repo)
├── requirements.txt           # Python dependencies
└── notebooks/
    └── ocr_mnist_training.ipynb # Notebook for model training
```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies:
  - streamlit
  - torch
  - torchvision
  - matplotlib
  - numpy
  - Pillow

## How to Run
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Ensure the pre-trained model file** `mnist_classifier_final.pth` is in the `models/` directory.
3. **Start the app:**
   ```sh
   streamlit run streamlit_app.py
   ```

## Notes
- The neural network architecture is defined in `src/network.py` and imported by the app.
- Model/data loading and utility functions are in `src/utils.py`.
- The app only uses the MNIST test set for visualization and does not retrain the model.
- To retrain or fine-tune the model, use the notebook in `notebooks/`.
