# Toward Reliable Pneumonia Detection:  
### An InternImage-Inspired CNN Framework with Grad-CAM Explainability

This repository contains the implementation of a lightweight, InternImage-Inspired Convolutional Neural Network designed for pneumonia detection using chest X-ray images. The project adapts key ideas from **InternImage (CVPR 2023)** such as hierarchical feature extraction and deformable-like spatial modeling—while remaining compatible with consumer-grade GPUs (e.g., RTX 4060 on Windows) that **cannot** compile official DCNv3 CUDA kernels.

The final model achieves **93.11% test accuracy**, outperforming a ResNet-18 baseline and an initial prototype while providing improved interpretability through **Grad-CAM**.

---

## 🚀 **Key Features**
- **InternImage-inspired architecture** using:
  - Hierarchical 4-stage feature extraction  
  - Patch embedding  
  - Depthwise convolution blocks simulating deformable behavior  
- Fully compatible with **Windows + CUDA** (no custom kernels needed)
- Strong performance on the **Chest X-ray Pneumonia Dataset**
- Complete training pipeline (baseline → V1 → V2 optimization)
- **Grad-CAM visualizations** for interpretability
- IEEE-style report included

---
