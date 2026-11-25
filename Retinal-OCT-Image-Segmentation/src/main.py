
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive

from preprocessing import preprocess_image
from segmentation import kmeans_segmentation_basic
from evaluation import calculate_iou, calculate_dice


# 1. Mount Google Drive

drive.mount('/content/drive')


# 2. Dataset Paths

scan_dir = '/content/drive/MyDrive/data/OCT_Dataset/img_dir/'
gt_dir = '/content/drive/MyDrive/data/OCT_Dataset/ann_dir/'


# 3. Segmentation Parameters

K = 4


# 4. Process OCT Images


files = sorted([
    f for f in os.listdir(scan_dir)
    if f.lower().endswith(('.tif', '.tiff'))
])

all_ious = []
all_dices = []

for image_name in files:

    print("Processing:", image_name)

    # Load grayscale OCT image
    img_gray = cv2.imread(
        os.path.join(scan_dir, image_name),
        cv2.IMREAD_GRAYSCALE
    )

    if img_gray is None:
        print("Could not load image:", image_name)
        continue

    # Preprocessing
    enhanced_img = preprocess_image(img_gray)

    # K-Means segmentation
    (
        cyst_mask_binary,
        segmented_img_colored,
        segmented_img_grayscale
    ) = kmeans_segmentation_basic(enhanced_img, K)

    # Load ground truth mask
    gt_path = os.path.join(gt_dir, image_name)

    gt_mask = cv2.imread(
        gt_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gt_mask is None:
        print("Ground truth missing:", image_name)
        continue

    # Resize ground truth (if necessary)
    if gt_mask.shape != cyst_mask_binary.shape:

        gt_mask = cv2.resize(
            gt_mask,
            (
                cyst_mask_binary.shape[1],
                cyst_mask_binary.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # Ground truth preprocessing
    if np.mean(gt_mask) < 128:
        gt_mask = cv2.bitwise_not(gt_mask)

    mask_gt = gt_mask > 128
    mask_pred = cyst_mask_binary > 0

    # Calculate evaluation metrics
    iou_score = calculate_iou(mask_pred, mask_gt)
    dice_score = calculate_dice(mask_pred, mask_gt)

    all_ious.append(iou_score)
    all_dices.append(dice_score)

    # 5. Visualization

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Original OCT')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(segmented_img_grayscale, cmap='gray')
    plt.title('K-Means Segmentation')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(mask_pred, cmap='gray')
    plt.title('Predicted Cyst Mask')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(mask_gt, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    plt.suptitle(
        f'{image_name} | IoU: {iou_score:.3f} | '
        f'Dice: {dice_score:.3f}'
    )

    plt.tight_layout()
    plt.show()


# 6. Average Evaluation Scores

if len(all_ious) > 0:

    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)

    print("\n" + "=" * 60)
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print("=" * 60)

else:
    print("No images were successfully evaluated.")


# 7. Comparative Histogram

if len(all_ious) > 0:

    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 20)

    plt.hist(
        all_ious,
        bins=bins,
        alpha=0.6,
        label='IoU Scores',
        edgecolor='black'
    )

    plt.hist(
        all_dices,
        bins=bins,
        alpha=0.6,
        label='Dice Scores',
        edgecolor='black'
    )

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Comparative Histogram of IoU and Dice Scores')

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
