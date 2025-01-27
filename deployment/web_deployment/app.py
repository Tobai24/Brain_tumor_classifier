import streamlit as st
import requests
from PIL import Image
import io


st.set_page_config(
    page_title="Brain Tumor Classifer 🧠",
    page_icon="🧠",
    layout="centered",
)


# Title of the Streamlit app
st.title("Brain Tumor Classifier 🧠")

st.markdown(
    """
    <style>
        @import url("https://fonts.googleapis.com/css2?family=LXGW+WenKai+TC&display=swap");

        /* Apply font globally */
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
            margin:20px auto;
            font-family: "LXGW WenKai TC", serif;
        }
        .stFileUploader:hover {
            border-color: #e91e63;
        }
        .stSpinner {
            color: #ff4081;
            font-size: 18px;
        }
        .stText, .stWrite, p, h2, h3, h4, h5, h6, li, span {
            font-size: 18px;
            line-height: 2;
            font-family: "LXGW WenKai TC", serif;
        }
        h1 {
            line-height: 2;
            font-family: "LXGW WenKai TC", serif;
            margin: 0px;
            padding: 0px;
        }
    </style>
    """, unsafe_allow_html=True
)

# Description
st.write("""
Upload an MRI image of the brain, and the model will predict whether it shows a **Glioma**, **Meningioma**, **Pituitary Tumor**, or a **normal brain MRI**.
""")

# File uploader for the image
uploaded_file = st.file_uploader("Upload a Brain MRI image...", type=["jpg", "jpeg", "png"])

# FastAPI endpoint URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

# If an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Make a POST request to the FastAPI /predict endpoint
    with st.spinner("Analyzing the image..."):
        try:
            response = requests.post(FASTAPI_URL, files={"file": image_bytes})
            if response.status_code == 200:
                result = response.json()
                st.success(result["message"])
            else:
                st.error(f"Error: {response.json().get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
