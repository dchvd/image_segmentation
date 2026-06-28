import os
from PIL import Image

class KagglePytorchService():

    def __init__(self, root_dir, image_set='train', transforms=None):
        self.transforms = transforms
        self.images_dir = os.path.join(root_dir, "JPEGImages")
        self.masks_dir = os.path.join(root_dir, "SegmentationClass")
        split_file = os.path.join(root_dir, "ImageSets", "Segmentation", f"{image_set}.txt")
        with open(split_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        img_path = os.path.join(self.images_dir, f"{img_name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{img_name}.png")
        
        image = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
            
        return image, target