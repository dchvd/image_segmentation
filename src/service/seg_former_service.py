import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


class SegFormerService:

    def __init__(self, num_classes=20, lr=6e-5, device=None,checkpoint="nvidia/segformer-b0-finetuned-ade-512-512"):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes  
        self.total_classes = self.num_classes + 1 
        self.lr = lr
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            checkpoint,
            num_labels=self.total_classes,
            ignore_mismatched_sizes=True,
        )
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

    @staticmethod
    def _prepare_masks(masks):
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        return masks.long()

    def _forward_upsampled(self, images):
        outputs = self.model(pixel_values=images)
        logits = outputs.logits  
        upsampled_logits = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_logits

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc="[Train SegFormer]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = self._prepare_masks(masks).to(self.device)

            self.optimizer.zero_grad()
            outputs = self._forward_upsampled(images)
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
            pbar = tqdm(val_loader, desc="[Val SegFormer]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = self._prepare_masks(masks).to(self.device)

                outputs = self._forward_upsampled(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=10, save_path="best_segformer_voc.pth"):
        """ Boucle principale d'entraînement """
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

    def get_segmentation_mask(self, image_tensor):
        self.model.eval()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self._forward_upsampled(image_tensor)  
            pred_mask = outputs.argmax(dim=1).squeeze(0)

        return pred_mask.cpu().numpy()

    def evaluate_segmentation(self, val_loader):
        print("\nCalculating MIoU for SegFormer...")
        self.model.eval()
        ious = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = self._prepare_masks(masks)

                outputs = self._forward_upsampled(images)
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