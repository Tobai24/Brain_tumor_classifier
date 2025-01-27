import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import torch.nn as nn
from collections import OrderedDict

# Initialize the Streamlit app
st.set_page_config(
    page_title="Brain Tumor Classifier 🧠",
    page_icon="🧠",
    layout="centered",
)

st.title("Brain Tumor Classifier 🧠")

st.markdown(
    """
    <style>
        @import url("https://fonts.googleapis.com/css2?family=LXGW+WenKai+TC&display=swap");

        html, body, [class*="css"] {
            font-family: "LXGW WenKai TC", serif;
            background-color: #f0f4f7;
            font-size: 20px;
        }

        .stButton button {
            background-color: #ff4081;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            height: 50px;
            width: 100%;
            cursor: pointer;
            font-family: "LXGW WenKai TC", serif;
        }
        .stButton button:hover {
            background-color: #e91e63;
        }
        .stFileUploader {
            background-color: #ffffff;
            border: 2px dashed #ff4081;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            font-family: "LXGW WenKai TC", serif;
        }
        .stFileUploader:hover {
            border-color: #e91e63;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    Upload an MRI image of the brain, and the model will predict whether it shows a **Glioma**, **Meningioma**, **Pituitary Tumor**, or a **normal brain MRI**.
    """
)

# File uploader for the image
uploaded_file = st.file_uploader("Upload a Brain MRI image...", type=["jpg", "jpeg", "png"])

# Define the custom image preprocessing class
class ConvertImage:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# Define the custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.model(x)

# Load the model checkpoint
checkpoint = torch.load("deployment/web_deployment/self_model.pth", map_location=torch.device("cpu"), weights_only=True)
new_state_dict = OrderedDict(
    {"model." + key: value for key, value in checkpoint["model_state_dict"].items()}
)

# Initialize the model
model = CustomCNN()
model.load_state_dict(new_state_dict)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    ConvertImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1855, 0.1856, 0.1856], std=[0.2003, 0.2003, 0.2003]),
])

# Process and predict if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    with st.spinner("Analyzing the image..."):
        try:
            # Preprocess the image
            input_tensor = preprocess(image).unsqueeze(0)
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_prob, predicted_class = torch.max(probabilities, 1)

            # Class mapping
            class_mapping = {0: "Glioma", 1: "Meningioma", 2: "Normal brain MRI", 3: "Pituitary Tumor"}
            prediction = class_mapping[predicted_class.item()]
            confidence = max_prob.item() * 100

            st.success(f"The model predicts that the image shows a **{prediction}** with a confidence of **{confidence:.2f}%**.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

