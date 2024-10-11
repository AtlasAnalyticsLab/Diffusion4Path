from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import io
import random

class KGHDataset(Dataset):
    """Dataset with tumor presence labels in text"""
    def __init__(self, root_dir, crop_size=None, p_uncond=0.0):
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.p_uncond = p_uncond
        self.image_paths = []
        self.labels = []

        # Iterate over all the subdirectories (folders) in root_dir
        for class_folder in sorted(self.root_dir.iterdir()):
            if class_folder.is_dir():
                # Use folder name as class label
                class_label = class_folder.name

                # Get all image files in the folder
                image_files = list(class_folder.glob("*.png"))  # Adjust if the images are in another format
                self.image_paths.extend(image_files)
                self.labels.extend([class_label] * len(image_files))
        
        # Shuffle the data
        combined = list(zip(self.image_paths, self.labels))
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

    def __len__(self):
        return len(self.image_paths)

    def get_random_crop(self, img, size):
        x = np.random.randint(0, img.width - size)
        y = np.random.randint(0, img.height - size)
        return img.crop((x, y, x + size, y + size))

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]


        image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
       
        if self.crop_size:
            image = self.get_random_crop(image, self.crop_size)

        image = np.array(image).astype(np.float32) / 127.5 - 1.0

        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()

        return {
            "image": image,
            "class_label": label,  
        }
