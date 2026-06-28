import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision.models.segmentation as segmentation_models


class FCNService:

    def __init__(self, num_classes=20, lr=1e-4, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes  
        self.total_classes = self.num_classes + 1  
        self.lr = lr

        print(f"[INFO] Utilisation de l'appareil : {self.device}")

       
        self.model = segmentation_models.fcn_resnet50(
            weights=segmentation_models.FCN_ResNet50_Weights.DEFAULT
        )

        # Remplacer la dernière couche du classifieur pour matcher nos classes
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, self.total_classes, kernel_size=1)

        self.model = self.model.to(self.device)

        # 2. Perte de segmentation standard : cross-entropy pixel par pixel
        # ignore_index=255 pour ignorer les pixels de contour (void) Pascal VOC
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    @staticmethod
    def _prepare_masks(masks):
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        return masks.long()

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc="[Train FCN]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = self._prepare_masks(masks).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)["out"] 
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Val FCN]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = self._prepare_masks(masks).to(self.device)

                outputs = self.model(images)["out"]
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=10, save_path="/results/best_fcn_voc.pth"):
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\n--- Époque {epoch + 1}/{epochs} ---")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            print(f"  [Train] Loss: {train_loss:.4f} | [Val] Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded weights : {path}")

    def get_segmentation_mask(self, image_tensor):
        self.model.eval()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)["out"]  
            pred_mask = output.argmax(dim=1).squeeze(0)  

        return pred_mask.cpu().numpy()

    def evaluate_segmentation(self, val_loader):
        self.model.eval()
        ious = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = self._prepare_masks(masks)

                outputs = self.model(images)["out"]
                preds = outputs.argmax(dim=1).cpu().numpy()
                true_masks = masks.numpy()

                for i in range(images.shape[0]):
                    pred_mask = preds[i]
                    true_mask = true_masks[i]
                    valid_mask = (true_mask != 255)

                    for cls in range(self.total_classes):
                        pred_cls = (pred_mask == cls)
                        target_cls = (true_mask == cls)

                        intersection = (pred_cls & target_cls & valid_mask).sum()
                        union = ((pred_cls | target_cls) & valid_mask).sum()

                        if union > 0:
                            ious.append(intersection / union)

        return sum(ious) / len(ious) if ious else 0.0