import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from transformers import SegformerForSemanticSegmentation

class ViTService:
    def __init__(self, model_name="nvidia/mit-b0", classes=21, lr=6e-5, device=None):
        """
        Service pour gérer un modèle Vision Transformer (SegFormer) sur Pascal VOC 2012.
        
        :param model_name: Modèle Hugging Face (ex: 'nvidia/mit-b0' à 'nvidia/mit-b5')
        :param classes: 20 objets + 1 background = 21
        :param lr: Taux d'apprentissage (les ViT aiment les LR plus faibles que CNN)
        :param device: Appareil cible ('cuda', 'cpu')
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = classes
        self.lr = lr
        
        print(f"[INFO] Utilisation de l'appareil : {self.device}")
        
        # 1. Chargement de SegFormer (ViT dédié à la segmentation)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=self.classes,
            ignore_mismatched_sizes=True # Permet d'adapter la tête de classification à 21 classes
        ).to(self.device)
        
        # 2. Fonction de perte (identique à U-Net pour comparaison juste)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 3. Optimiseur : AdamW est fortement recommandé pour les Transformers
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
    def _calculate_batch_miou(self, preds, targets):
        """ Calcule le mIoU en ignorant la valeur 255 """
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
        """ Entraîne le ViT sur une époque """
        self.model.train()
        running_loss = 0.0
        running_miou = 0.0
        
        pbar = tqdm(train_loader, desc="[Train ViT]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
                
            self.optimizer.zero_grad()
            
            # Forward pass du Transformer
            outputs = self.model(pixel_values=images)
            logits = outputs.logits # Forme brute : [B, 21, H/4, W/4]
            
            # Subtilité ViT : Interpolation pour revenir à la taille originale du masque [H, W]
            upsampled_logits = nn.functional.interpolate(
                logits, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            loss = self.criterion(upsampled_logits, masks)
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            running_loss += loss.item()
            preds = torch.argmax(upsampled_logits, dim=1)
            running_miou += self._calculate_batch_miou(preds, masks)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return running_loss / len(train_loader), running_miou / len(train_loader)

    def validate_epoch(self, val_loader):
        """ Évalue le ViT sur le jeu de validation """
        self.model.eval()
        running_loss = 0.0
        running_miou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Val ViT]")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                if masks.dim() == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                # Interpolation à la taille réelle
                upsampled_logits = nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                
                loss = self.criterion(upsampled_logits, masks)
                running_loss += loss.item()
                
                preds = torch.argmax(upsampled_logits, dim=1)
                running_miou += self._calculate_batch_miou(preds, masks)
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        return running_loss / len(val_loader), running_miou / len(val_loader)

    def fit(self, train_loader, val_loader, epochs=10, save_path="/results/best_vit_voc.pth"):
        """ Boucle principale d'entraînement """
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
                # Sauvegarde spécifique pour les modèles Hugging Face
                self.model.save_pretrained(save_path)

    def load_weights(self, path):
        """ Charge le modèle sauvegardé via save_pretrained """
        self.model = SegformerForSemanticSegmentation.from_pretrained(path).to(self.device)
        print(f"[INFO] Modèle ViT chargé depuis {path}")

    def predict(self, image_tensor):
        """ Prédiction sur une image seule [3, H, W] """
        self.model.eval()
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            outputs = self.model(pixel_values=image_tensor)
            # Interpolation à la taille originale de l'image d'entrée
            upsampled_logits = nn.functional.interpolate(
                outputs.logits, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            preds = torch.argmax(upsampled_logits, dim=1)
        return preds.squeeze(0).cpu().numpy()