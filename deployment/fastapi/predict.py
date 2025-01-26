from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io

# Initialize the FastAPI app
app = FastAPI()

#the class for converting the image to RGB
class ConvertImage:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img   

# Load the pre-trained PyTorch model
model = torch.load("des_model.pth")
model.eval()

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    ConvertImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.1855, 0.1856, 0.1856], std=[0.2003, 0.2003, 0.2003])
])

# Define the prediction function
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Check if a GPU is available and move the input batch to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch = input_batch.to(device)
    model.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)
        
    probabilities = torch.nn.functional.softmax(output, dim=1)
    
    # Get the predicted class (assuming softmax or similar)
    _, predicted_class = torch.max(probabilities, 1)

    # Return the prediction
    return {"prediction": predicted_class.item(), "probabilities": probabilities.cpu().numpy().tolist()}

# Run the app if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
