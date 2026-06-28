import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

class TransformationService():

    def __init__(self, size=(256, 256)):
        self.size = size
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, target):
        image = TF.resize(image, self.size, interpolation=T.InterpolationMode.BILINEAR)
        target = TF.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        image = self.img_transform(image)
        target = torch.from_numpy(np.array(target)).long()
        
        return image, target