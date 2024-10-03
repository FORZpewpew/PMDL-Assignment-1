import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

script_dir = os.path.dirname(__file__)

dataset_path = os.path.join(script_dir, "..", "datasets", "cat")


print(f"cuda: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_grayscale = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to Tensor
])

transform_color = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert to Tensor
])

class ColorToGrayscaleDataset(Dataset):
    def __init__(self, data_dir, transform_grayscale=None, transform_color=None):
        self.data_dir = data_dir
        self.transform_grayscale = transform_grayscale
        self.transform_color = transform_color
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path)

        # Convert to grayscale for input
        grayscale_image = self.transform_grayscale(image)
        # Original image for output
        color_image = self.transform_color(image)

        return grayscale_image, color_image

dataset = ColorToGrayscaleDataset(dataset_path, transform_grayscale, transform_color)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

model = ColorizationCNN().to(device)
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode

    running_loss = 0.0
    for grayscale_images, color_images in data_loader:
        # Move data to the GPU
        grayscale_images = grayscale_images.to(device)
        color_images = color_images.to(device)

        optimizer.zero_grad()

        outputs = model(grayscale_images)
        loss = criterion(outputs, color_images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}')

save_path = os.path.join(script_dir, "..", "deployment", "api", "colorization_cnn.pth")

torch.save(model.state_dict(), save_path)