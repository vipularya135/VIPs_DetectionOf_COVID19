import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import os

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Use Tanh for generator output
        )

    def forward(self, z):
        return self.model(z)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Output a probability
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
batch_size = 32
epochs = 10
latent_dim = 100
learning_rate = 1e-3

# Paths
input_path = "Covid"  # Path to CT scan images
output_path = "output"  # Path to save generated images
os.makedirs(output_path, exist_ok=True)

# Data loader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Grayscale input for CT scan
    transforms.Resize((64, 64)),  # Resize images to 64x64 for DCGAN
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Custom Dataset for Images Only
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# Instantiate the dataset and data loader
dataset = ImageDataset(root_dir=input_path, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize the generator and discriminator, and send to device
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Initialize optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for batch_idx, real_images in enumerate(data_loader):
        real_images = real_images.to(device)

        # Train Discriminator
        optimizer_d.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim, 1, 1).to(device)  # Random noise
        fake_images = generator(z)  # Generate fake images

        # Calculate the loss for the discriminator
        d_loss_real = nn.functional.binary_cross_entropy(discriminator(real_images), torch.ones(real_images.size(0), 1).to(device))
        d_loss_fake = nn.functional.binary_cross_entropy(discriminator(fake_images.detach()), torch.zeros(real_images.size(0), 1).to(device))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        g_loss = nn.functional.binary_cross_entropy(discriminator(fake_images), torch.ones(real_images.size(0), 1).to(device))
        g_loss.backward()
        optimizer_g.step()

    print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Generate and save images
def generate_and_save_images(generator, num_images, output_dir):
    generator.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, latent_dim, 1, 1).to(device)  # Sample from latent space
            generated_image = generator(z)  # Generate fake image
            
            # Rescale to [0, 1]
            generated_image = (generated_image + 1) / 2
            
            # Save the generated image
            output_file_path = os.path.join(output_dir, f'synthetic_ct_scan_{i+1}.png')
            save_image(generated_image, output_file_path)

            print(f"Generated and saved: {output_file_path}")

# Generate and save 200 lung CT scan images
generate_and_save_images(generator, num_images=200, output_dir=output_path)

print(f"Successfully generated 200 CT scan images at {output_path}")
