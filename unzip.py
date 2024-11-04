import zipfile
import os

# Define the path to the zip file and the extraction directory
zip_file_path = 'output.zip'
extraction_path = 'output'

# Create the extraction directory if it doesn't exist
os.makedirs(extraction_path, exist_ok=True)

# Unzip the folder
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# List the files in the extracted directory
extracted_files = os.listdir(extraction_path)
extracted_files
