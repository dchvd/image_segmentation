import os
import numpy as np
import matplotlib.pyplot as plt


class PerformanceVisualizerService():

    DEFAULT_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#E99CDC"]

    def __init__(self, output_dir="results/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def save(self, save_name):
        path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved visualization : {path}")
        return path

    def plot_miou_comparison(self, results, save_name="miou_comparison.png"):
        names = list(results.keys())
        values = [v * 100 if v <= 1 else v for v in results.values()]

        fig, ax = plt.subplots(figsize=(7, 5))
        colors = self.DEFAULT_COLORS[:len(names)]
        bars = ax.bar(names, values, color=colors)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.2f}%", ha="center", va="bottom", fontsize=11)

        ax.set_ylabel("mIoU (%)")
        ax.set_title("Comparaison du mIoU sur l'ensemble de validation")
        ax.set_ylim(0, max(values) * 1.25)
        plt.xticks(rotation=15)

        return self.save(save_name)

    def plot_loss_curves(self, history, save_name="loss_curves.png"):
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, (model_name, losses) in enumerate(history.items()):
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            epochs = range(1, len(losses["train"]) + 1)
            ax.plot(epochs, losses["train"], linestyle="--", color=color,
                     label=f"{model_name} (train)")
            ax.plot(epochs, losses["val"], linestyle="-", color=color,
                     label=f"{model_name} (val)")

        ax.set_xlabel("Époque")
        ax.set_ylabel("Loss")
        ax.set_title("Courbes d'apprentissage par modèle")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        return self.save(save_name)

    def plot_qualitative_comparison(self, image, true_mask, predictions, save_name="qualitative_comparison.png"):

        n_cols = 2 + len(predictions)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        axes[0].imshow(image)
        axes[0].set_title("Image originale")
        axes[0].axis("off")

        axes[1].imshow(self._mask_to_rgb(true_mask))
        axes[1].set_title("Masque réel")
        axes[1].axis("off")

        for i, (model_name, pred_mask) in enumerate(predictions.items()):
            axes[2 + i].imshow(self._mask_to_rgb(pred_mask))
            axes[2 + i].set_title(model_name)
            axes[2 + i].axis("off")

        return self.save(save_name)

    def plot_per_class_iou(self, per_class_results, class_names, save_name="per_class_iou.png"):

        n_models = len(per_class_results)
        n_classes = len(class_names)
        x = np.arange(n_classes)
        width = 0.8 / n_models

        fig, ax = plt.subplots(figsize=(max(10, n_classes * 0.6), 5))

        for i, (model_name, ious) in enumerate(per_class_results.items()):
            offset = (i - n_models / 2) * width + width / 2
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            ax.bar(x + offset, np.array(ious) * 100, width, label=model_name, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("IoU (%)")
        ax.set_title("IoU par classe et par modèle")
        ax.legend()

        return self.save(save_name)