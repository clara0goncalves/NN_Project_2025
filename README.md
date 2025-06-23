# Water Body Segmentation Project

Authors:
De Salvo Ludovico
Gonçalves Clara

## Introduction

This project provides a complete and streamlined pipeline for segmentation of water bodies from satellite using PyTorch. It is built to be flexible and powerful, allowing for easy training, evaluation, and inference with a selection of advanced deep learning models.

## Features

This project is packed with features to support a robust deep learning workflow from start to finish:

-   **Model Selection**: Easily train and evaluate a selection of powerful segmentation models:
    -   AER U-Net
    -   U-Net++ (with a pre-trained EfficientNet-B4 encoder)
    -   SegFormer-B4
-   **Advanced Training**:
    -   **Loss Functions**: A choice of multiple loss functions to tackle class imbalance, including `FocalLovaszLoss` (default), `DiceLoss`, `FocalLoss`, `TverskyLoss`, `BCEWithLogitsLoss`, and a combined BCE+Dice loss.
    -   **Optimizers & Schedulers**: Supports `Adam`, `AdamW`, and `SGD` optimizers with learning rate schedulers like `ReduceLROnPlateau` and `CosineAnnealingLR`.
    -   **Modern Training Techniques**: Key features are enabled by default through the interactive menu to boost performance and speed:
        -   Automatic Mixed Precision (AMP) for faster training on compatible GPUs.
        -   Early Stopping to prevent overfitting and save time.
        -   Gradient Clipping to stabilize training.
-   **Comprehensive Evaluation**:
    -   Calculates a full suite of segmentation metrics: `IoU (Jaccard)`, `Dice Score`, `Pixel Accuracy`, `Precision`, `Recall`, and `F1-Score`.
    -   **Ensemble Evaluation**: Evaluate the combined performance of two different models by averaging their predictions.
    -   **Test-Time Augmentation (TTA)**: Improve prediction accuracy by averaging predictions over multiple augmented versions of a test image.
    -   **Interactive Data Verification**: During ensemble evaluation, the script can automatically flag samples with a low IoU score, display the ground truth mask to you, and allow you to discard the sample from the final metrics if the mask is broken.
-   **Logging & Visualization**:
    -   Logs all training and validation metrics to **TensorBoard** for real-time monitoring.
    -   Saves a CSV file (`training_history.csv`) and a plot (`training_plots.png`) of the training session in the experiment's checkpoint directory.
-   **Usability**:
    -   **Interactive CLI**: A simple and powerful command-line menu (`main.py`) guides you through every step of the process.

## Project Structure

The project is organized into a standard structure for clarity and scalability.

```bash
NN_Project_2025/
│
├── checkpoints/
│   └── <experiment_name>     # Stores trained model weights (.pth), config, and training history
│
├── datasets/
│   ├── train/                 # Training images and masks (created by prepare_data.py)
│   ├── val/                   # Validation images and masks
│   └── test/                  # Test images and masks
│
├── evaluation_results/
│   └── <evaluation_run_name>/ # Stores prediction examples and metrics from evaluate.py
│
├── models/
│   ├── aer_unet.py            # Defines the AER U-Net architecture
│   └── segformer.py           # Defines the SegFormer architecture
│
├── utils/
│   ├── data_utils.py          # Defines the PyTorch Dataset class
│   ├── losses.py              # Contains all custom loss function implementations
│   ├── metrics.py             # Contains all evaluation metric implementations
│   ├── prepare_data.py        # Script to split the raw dataset into train/val/test sets
│   └── visualize_dataset.py   # Script to view a sample batch of data
│
├── Water Bodies Dataset/
│   ├── Images/                # Your raw source images go here
│   └── Masks/                 # Your raw source masks go here
│
├── evaluate.py                # Main script for evaluating trained models
├── main.py                    # The main interactive menu to run the project
├── train.py                   # The main script for training new models
└── requirements.txt           # All Python package dependencies
```

## How to Run the Project

The entire workflow is managed through the interactive `main.py` script.

**1. Setup**

-   Clone the repository to your local machine.
-   Place your raw dataset into the `Water Bodies Dataset/` folder, with images in `Images/` and masks in `Masks/`.

**2. Launch the Interactive Menu**

Open your terminal in the project's root directory and run:

```bash
python3 main.py
```

This will bring up the main menu:

```bash
============================
  Water Segmentation Menu
============================
1. Install/Update Requirements
2. Prepare Data (Split Raw Dataset)
3. Visualize a Batch of the Dataset
4. Train a Model
5. Evaluate a Model
6. Exit
```

### Workflow Steps

1.  **Install Requirements (Option 1)**: The first time you run the project, select this option. It will install all necessary Python packages from the `requirements.txt` file.

2.  **Prepare Data (Option 2)**: This executes the `utils/prepare_data.py` script. It takes your raw data from the `Water Bodies Dataset/` folder and automatically splits it into `train`, `val`, and `test` sets within the `datasets/` directory. You only need to run this step once.

3.  **Visualize Dataset (Option 3)**: This is a helpful utility to view a sample batch of images and masks from your training data. It's useful for verifying that your data is being loaded and augmented correctly before starting a long training session.

4.  **Train a Model (Option 4)**: This is the primary training step. The script will interactively guide you through:
    * Selecting a model architecture (`AER U-Net`, `U-Net++`, `SegFormer-B4`).
    * Setting training parameters (e.g., learning rate, batch size).
    * Choosing a loss function from the available options.
    * Confirming advanced training options like AMP and Early Stopping.

5.  **Evaluate a Model (Option 5)**: After a model has been trained, use this option to test its performance on the test set. The evaluation script allows you to:
    * Choose between evaluating a single model or an ensemble of two models.
    * Select the specific trained model checkpoint(s) you want to test.
    * Enable optional features like Test-Time Augmentation (TTA) for improved accuracy.
    * Interactively verify and filter out potentially broken ground truth masks during ensemble evaluation.

## Results

```bash
===================================================
EVALUATION RESULTS
===================================================

Dice Score:
Mean: 0.9021 +- 0.1079
Range: [0.1597, 1.0000]

Iou Score:
Mean: 0.8361 +- 0.1499
Range: [0.0868, 1.0000]

Pixel acc Score:
Mean: 0.9560 +- 0.0505
Range: [0.5815, 1.0000]

Precision Score:
Mean: 0.9241 +- 0.0937
Range: [0.3489, 1.0000]

Recall Score:
Mean: 0.8944 +- 0.1318
Range: [0.0890, 1.0000]

F1 Score:
Mean: 0.9021 +- 0.1079
Range: [0.1597, 1.0000]

Overall Dataset Metrics (from total counts):
Precision: 0.9466
Recall (Sensitivity): 0.9183
F1-Score: 0.9322
```
These are excellent results overall. The high mean scores for IoU (0.836) and Dice (0.902) suggest the model is highly effective and accurate on the majority of the data. However, the wide performance range and high standard deviation point to a small subset of challenging images where the model fails, which could be an area to target for future improvement.

## References

* **AER U-Net**: Naga Surekha Jonnala1, Shaik Siraaj, Y. Prastuti, P. Chinnababu, B. Praveen babu, Shonak Bansal, Prashant Upadhyaya, Krishna Prakash1, Mohammad Rashed Iqbal Faruque & K. S. Al-mugren (2024). AER U-Net: attention-enhanced multi-scale residual U-Net structure for water body segmentation using Sentinel-2 satellite images 

* **Dataset**: Satellite Images of Water Bodies [(Kaggle)](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)

* **SegFormer**: Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. 

* **Lovász-Softmax Loss**: Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks. 

