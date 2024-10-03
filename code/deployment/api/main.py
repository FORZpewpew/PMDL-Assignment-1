from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse  
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import io

# Define your model architecture (example for a simple CNN)
import torch.nn as nn

class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Grayscale input
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)  # Output 3 channels for RGB

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        
        # Decoder
        x = F.relu(self.deconv1(x))
        x = F.interpolate(x, scale_factor=2)  # Upsample
        x = F.relu(self.deconv2(x))
        x = F.interpolate(x, scale_factor=2)  # Upsample
        x = self.deconv3(x)
        
        return x

app = FastAPI()

# Initialize and load the model
model = ColorizationCNN()
model.load_state_dict(torch.load("colorization_cnn.pth", map_location=torch.device('cpu')))
model.eval() 
# Define any necessary image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the grayscale image file directly
    image = Image.open(file.file).convert("L")  # Ensure the image is in grayscale
    input_data = transform(image).unsqueeze(0)

    # Make prediction using the model
    with torch.no_grad():
        output_data = model(input_data)

    # Assuming output_data is of shape (1, 3, height, width)
    output_data = output_data.squeeze(0)  # Remove batch dimension
    # The shape is now (3, height, width), need to permute it
    output_data = output_data.permute(1, 2, 0)  # Change to (height, width, 3)

    # Ensure the output is scaled correctly and convert to uint8
    output_image_data = (output_data.cpu().numpy() * 255).astype('uint8')  # Scale to 0-255

    output_image = Image.fromarray(output_image_data)

    output_image_io = io.BytesIO()
    output_image.save(output_image_io, format='PNG')
    output_image_io.seek(0)

    # Return the output image as a StreamingResponse
    return StreamingResponse(output_image_io, media_type="image/png")