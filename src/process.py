import os
import torch
from torch.utils.data import DataLoader

from src.service.unet_service import UNetService
from src.service.fcn_service import FCNService
from src.service.vit_service import ViTService
from src.service.transformation_service import TransformationService
from src.service.kaggle_pytorch_service import KagglePytorchService


def process():
    DATA_DIR = "data/VOC2012_train_val" 
    # BATCH_SIZE = 16
    # EPOCHS = 10
    # IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 5
    IMAGE_SIZE = (128, 128)
    DEVICE = "cpu"

    transform = TransformationService(size=IMAGE_SIZE)

    train_dataset = KagglePytorchService(root_dir=DATA_DIR, image_set='train', transforms=transform)
    val_dataset = KagglePytorchService(root_dir=DATA_DIR, image_set='val', transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    results = {}


    print("Starting training for U-Net ... ")
    unet_service = UNetService(encoder_name="resnet34", classes=21, lr=1e-4, device=DEVICE)
    unet_checkpoint_path = os.path.join("results", "best_unet.pth")

    if not os.path.exists(unet_checkpoint_path):
        print("Starting training for U-Net ... ")
        unet_service.fit(train_loader, val_loader, epochs=EPOCHS, save_path=unet_checkpoint_path)
    else:
        print(f"This model was already trained and saved at ({unet_checkpoint_path}). Ignored.")
        unet_service.load_weights(unet_checkpoint_path)

    _, best_unet_miou = unet_service.validate_epoch(val_loader)
    results["U-Net (ResNet34)"] = best_unet_miou


    print("Starting training for FCN ...")
    fcn_service = FCNService(device=DEVICE)
    fcn_checkpoint_path = os.path.join("results", "best_fcn.pth")

    if not os.path.exists(fcn_checkpoint_path):
        print("Starting training for FCN ...")
        fcn_service.fit(train_loader, val_loader, epochs=EPOCHS, save_path=fcn_checkpoint_path)
    else:
        print(f"This model was already trained and saved at {fcn_checkpoint_path}). Ignored.")
        fcn_service.load_weights(fcn_checkpoint_path)
    
    fcn_miou = fcn_service.evaluate_segmentation(val_loader)
    results["FCN (ResNet50)"] = fcn_miou
    print(DEVICE)


    print("Starting training for ViT ...")
    vit_service = ViTService(model_name="nvidia/mit-b0", classes=21, lr=6e-5, device=DEVICE)
    vit_service.fit(train_loader, val_loader, epochs=EPOCHS, save_path="best_vit_segformer")
    _, best_vit_miou = vit_service.validate_epoch(val_loader)
    results["ViT (SegFormer B0)"] = best_vit_miou

    print("\n\n" + "📊" + " ="*15 + " BILAN COMPARATIF " + "= "*15 + "\n")
    print(f"{'Architecture / Approche':<30} | {'Mean IoU (mIoU) sur Validation':<35}")
    print("-" * 70)
    for model_name, miou_score in results.items():
        print(f"{model_name:<30} | {miou_score * 100:.2f} %")
    print("-" * 70)