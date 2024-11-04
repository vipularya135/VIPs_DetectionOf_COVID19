import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16_bn
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
import os

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define VAE with pretrained VGG encoder
class PretrainedVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(PretrainedVAE, self).__init__()
        # Load a pretrained VGG16 as the encoder
        pretrained_vgg = vgg16_bn(pretrained=True).features
        
        # Encoder using VGG layers
        self.encoder = nn.Sequential(
            pretrained_vgg,
            nn.Flatten(),
        )
        
        self.fc_mu = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(512 * 7 * 7, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 512, 7, 7)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Hyperparameters
batch_size = 32
epochs = 10
latent_dim = 20
learning_rate = 1e-3

# Paths
input_path = "Covid"  # Path to CT scan images
output_path = "output"  # Path to save generated images
os.makedirs(output_path, exist_ok=True)

# Data loader
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB (3 channels)
    transforms.Resize((512, 512)),
    transforms.ToTensor()
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

# Initialize the pretrained VAE model, optimizer, and send to device
model = PretrainedVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(data_loader.dataset):.4f}")

# Generate and save images
def generate_and_save_images(model, num_images, output_dir):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i in range(num_images):
            z = torch.randn(1, latent_dim).to(device)  # Sample from latent space
            generated_image = model.decode(z)  # Decode the latent vector
            
            # Convert to grayscale and resize to (512x512)
            generated_image = generated_image.cpu().squeeze(0)
            
            # Save the generated image
            output_file_path = os.path.join(output_dir, f'synthetic_ct_scan_{i+1}.png')
            save_image(generated_image, output_file_path)

            print(f"Generated and saved: {output_file_path}")

# Generate and save 200 lung CT scan images
generate_and_save_images(model, num_images=200, output_dir=output_path)

print(f"Successfully generated 200 CT scan images at {output_path}")
