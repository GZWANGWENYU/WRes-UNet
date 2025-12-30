# WRESUNET - Weighted Residual UNet for ICH Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-orange)

This repository implements **WRESUNET**, a weighted residual UNet designed for **intracerebral hemorrhage (ICH) segmentation** on CT images. The model integrates residual connections and UNet architecture for improved segmentation performance.

---

## 🧠 Features

- **WRes-UNet Architecture**: Combines residual connections with UNet for stable training and better feature propagation.
- **Multi-metric Evaluation**: Computes IoU, Dice, F1, HD95, Precision, Recall, Specificity.
- **Training & Testing Visualization**: Saves segmented images during training and testing.
- **Model Complexity Info**: Automatically calculates FLOPs and parameter count.

---

## 📦 Requirements

```bash
Python >= 3.8
PyTorch >= 1.8
torchvision
tqdm
pandas
matplotlib
scikit-learn
tensorboard
thop
