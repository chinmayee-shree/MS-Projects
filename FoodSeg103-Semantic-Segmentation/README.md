# Food Image Segmentation Using Deep Learning

## Overview

This project explores semantic segmentation of food images using deep learning. It compares three segmentation architectures on the **FoodSeg103 dataset**, which contains 103 food categories plus a background class (104 classes in total).

The objective is to identify and segment different food regions at the pixel level and compare the performance of CNN-based and transformer-based approaches.

## Models Implemented

Three models are implemented and evaluated:

1. **Baseline U-Net**
   - U-Net architecture with a pretrained ResNet34 encoder.
   - Uses ImageNet pretrained weights.
   - Serves as the baseline segmentation model.

2. **Improved CNN – DeepLabV3+**
   - DeepLabV3+ architecture with a pretrained ResNet50 encoder.
   - Uses data augmentation to improve generalization.
   - Applies horizontal-flip test-time augmentation (TTA).

3. **Transformer – SegFormer-B2**
   - Transformer-based semantic segmentation model.
   - Initialized from the pretrained `nvidia/segformer-b2-finetuned-ade-512-512` checkpoint.
   - Fine-tuned for the 104 FoodSeg103 classes.
   - Uses data augmentation and test-time augmentation.

## Dataset

**FoodSeg103** is a food image segmentation dataset containing images and pixel-level segmentation masks.

- Dataset: FoodSeg103
- Number of classes: 104, including background
- Training images: 4,983
- Test images: 2,135
- Validation set: 10% of the original training data

The dataset is downloaded and accessed through Google Drive in the notebook. Dataset files are not included in this repository.

## Methodology

The project follows these main steps:

1. Load and prepare the FoodSeg103 dataset.
2. Split the training data into training and validation sets.
3. Resize images and masks to the input dimensions required by each model.
4. Apply preprocessing and data augmentation where appropriate.
5. Train the three segmentation models.
6. Evaluate model predictions using pixel-level segmentation metrics.
7. Visualize predicted masks alongside input images and ground-truth masks.

## Evaluation Metrics

The models are evaluated using:

- **Pixel Accuracy:** Percentage of correctly classified pixels.
- **Mean Intersection over Union (mIoU):** Measures overlap between predicted and ground-truth segmentation regions.
- **Mean Dice Score:** Measures similarity between predicted and ground-truth masks.

## Results

The SegFormer-B2 experiment achieved:

| Metric | Test Result |
|---|---:|
| Best Validation mIoU | 0.4073 |
| Test mIoU | 0.3544 |
| Best Validation Epoch | 28 |

The notebook also contains experiments with the baseline U-Net and DeepLabV3+. Their results can be added here once the corresponding evaluation outputs are collected.

### Single-Image Demonstration

For one demonstrated image, the segmentation evaluation produced:

| Metric | Result |
|---|---:|
| Pixel Accuracy | 0.9857 |
| Mean IoU | 0.9577 |
| Mean Dice | 0.9783 |

These values are for a single-image demonstration and should not be interpreted as overall test-set performance.

## Technologies Used

- Python
- PyTorch
- Torchvision
- Segmentation Models PyTorch
- Hugging Face Transformers
- Albumentations
- OpenCV
- NumPy
- Matplotlib
- Google Colab

## Repository Structure

```text
Image-Segmentation/
├── notebooks/
│   └── Segmentation.ipynb
├── results/
│   ├── figures/
│   └── metrics/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Image-Segmentation
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook in Google Colab or Jupyter Notebook.

**Note:** The dataset and pretrained model checkpoints must be downloaded separately. The notebook includes the relevant loading and setup steps.

## Limitations

- Food segmentation can be challenging when food items overlap or have similar visual appearances.
- Performance can vary across food categories.
- The three architectures use different input resolutions and training configurations.
- The single-image demonstration metrics do not represent overall model performance.

## Future Improvements

- Perform a more extensive hyperparameter search.
- Compare all three models using the same evaluation protocol.
- Explore additional transformer architectures.
- Improve performance on visually similar or underrepresented food categories.
- Add more qualitative segmentation visualizations.

## Author

Developed as a deep learning project exploring semantic segmentation and comparing CNN-based and transformer-based architectures.
