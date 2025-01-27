import requests

# Define the URL of the FastAPI app
url = "http://127.0.0.1:8000/predict"

# Path to the sample image you want to test
sample_image_path = "image.png"

# Open the image file in binary mode
with open(sample_image_path, "rb") as image_file:
    # Send a POST request to the /predict endpoint
    response = requests.post(url, files={"file": image_file})

# Check if the request was successful
if response.status_code == 200:
    # Print the prediction and probabilities
    result = response.json()
    print(result["message"])

else:
    # Print the error message
    print("Error:", response.json())