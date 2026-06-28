import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp

class UNetService:
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=21, lr=1e-4, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.lr = lr
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self.classes
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def _calculate_batch_miou(self, preds, targets):
        ious = []
        for cls in range(self.classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)
            valid_mask = (targets != 255)
            intersection = (pred_cls & target_cls & valid_mask).sum().item()
            union = ((pred_cls | target_cls) & valid_mask).sum().item()
            if union == 0:
                continue
            ious.append(intersection / union)
            
        return sum(ious) / len(ious) if ious else 0.0

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        running_miou = 0.0
        
        pbar = tqdm(train_loader, desc="[Train]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            running_miou += self._calculate_batch_miou(preds, masks)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_miou = running_miou / len(train_loader)
        return epoch_loss, epoch_miou

    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        running_miou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Val]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                running_miou += self._calculate_batch_miou(preds, masks)
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        epoch_loss = running_loss / len(val_loader)
        epoch_miou = running_miou / len(val_loader)
        return epoch_loss, epoch_miou

    def fit(self, train_loader, val_loader, epochs=10, save_path="/results/best_unet_voc.pth"):
        best_val_miou = 0.0
        
        for epoch in range(epochs):
            print(f"\n--- Époque {epoch + 1}/{epochs} ---")
            
            train_loss, train_miou = self.train_epoch(train_loader)
            val_loss, val_miou = self.validate_epoch(val_loader)
            
            print(f"Époque {epoch + 1} Terminée :")
            print(f"  [Train] Loss: {train_loss:.4f} | mIoU: {train_miou:.4f}")
            print(f"  [Val]   Loss: {val_loss:.4f} | mIoU: {val_miou:.4f}")
            
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(self.model.state_dict(), save_path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded weights : {path}")

    def predict(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0) 
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            preds = torch.argmax(outputs, dim=1)
        return preds.squeeze(0).cpu().numpy() 