from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io
import uvicorn
import torch.nn as nn
from collections import OrderedDict

# Initialize the FastAPI app
app = FastAPI()

class ConvertImage:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img   

# Define the custom model class
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.model = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Convolutional Layer 2
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Convolutional Layer 3
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # Flatten Layer
            nn.Flatten(),
            # Fully Connected Layers
            nn.Linear(in_features=16 * 28 * 28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=4)
        )

    def forward(self, x):
        return self.model(x)

# Load the checkpoint
checkpoint = torch.load("self_model.pth", map_location=torch.device('cpu'))

# Add the "model." prefix to the keys in the state_dict
new_state_dict = OrderedDict()
for key, value in checkpoint["model_state_dict"].items():
    new_key = "model." + key  # Add the "model." prefix
    new_state_dict[new_key] = value

# Initialize the model
model = CustomCNN()
model.load_state_dict(new_state_dict)
model.eval()

# The image preprocessing pipeline
preprocess = transforms.Compose([
    ConvertImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.1855, 0.1856, 0.1856], std=[0.2003, 0.2003, 0.2003])
])

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # Make the prediction
        with torch.no_grad():
            output = model(input_batch)
            
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get the predicted class
        _, predicted_class = torch.max(probabilities, 1)
        
        class_mapping = {0: 'glioma', 1: 'meningioma', 2: 'notumor', 3: 'pituitary'}
        pred = class_mapping[predicted_class.item()]

        # Return the prediction
        return {"prediction": pred, "probabilities": probabilities.cpu().numpy().tolist()}

    except Exception as e:
        return {"error": str(e)}

# Run the app if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)