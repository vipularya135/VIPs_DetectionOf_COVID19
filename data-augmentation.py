import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Define paths
input_path_covid = r"C:\Users\DELL\Desktop\covid19\Covid"
input_path_normal = r"C:\Users\DELL\Desktop\covid19\Normal"
output_path_covid = r"C:\Users\DELL\Desktop\covid19\Augmented\Covid"
output_path_normal = r"C:\Users\DELL\Desktop\covid19\Augmented\Normal"

# Create output directories if they don't exist
os.makedirs(output_path_covid, exist_ok=True)
os.makedirs(output_path_normal, exist_ok=True)

# Define transformations
augmentations = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.RandomAffine(degrees=0, shear=10)
])

def augment_and_save_images(input_folder, output_folder, augmentations, num_augmentations=4):
    # Loop through each image in the folder
    for img_name in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        
        try:
            img = Image.open(img_path).convert("RGB")  # Open image and convert to RGB

            # Apply each augmentation and save
            for i in range(num_augmentations):
                augmented_img = augmentations(img)  # Apply augmentations
                augmented_img_path = os.path.join(output_folder, f"{os.path.splitext(img_name)[0]}_aug_{i}.png")
                augmented_img.save(augmented_img_path)  # Save augmented image
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

# Apply augmentations and save in output folders
print("Augmenting Covid-19 images...")
augment_and_save_images(input_path_covid, output_path_covid, augmentations)

print("Augmenting Normal images...")
augment_and_save_images(input_path_normal, output_path_normal, augmentations)

print("Augmentation complete.")
