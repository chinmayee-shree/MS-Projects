# Running the Retinal OCT Segmentation Project

## Environment

The project was developed and executed using Google Colab.

## Requirements

The project uses the following Python libraries:

- NumPy
- OpenCV
- Matplotlib

## Dataset Setup

The code expects the OCT dataset to be stored in Google Drive with the following folder structure:

```text
MyDrive/
└── data/
    └── OCT_Dataset/
        ├── img_dir/
        └── ann_dir/
```

- `img_dir/` contains the original OCT images.
- `ann_dir/` contains the corresponding ground-truth masks.

## Running the Notebook

1. Open the project notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Ensure that the dataset is stored in the expected directory.
4. Run the notebook cells in order.
5. Review the segmentation outputs and evaluation metrics.

## Evaluation

The project calculates:

- Intersection over Union (IoU)
- Dice Similarity Coefficient

The initial experiment evaluated 10 images.

## Important Note

The current implementation uses classical image processing and K-means clustering. It is an experimental project and is not intended for clinical diagnosis.
