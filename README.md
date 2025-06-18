# Water Body Segmentation with U-Net

This project implements a U-Net architecture for semantic segmentation of water bodies in satellite/aerial imagery.

## Features

- Complete U-Net implementation with PyTorch
- Multiple loss functions (BCE, Dice, Combined, Focal, Tversky)
- Comprehensive evaluation metrics
- Data augmentation support
- Tensorboard logging
- Easy-to-use inference utilities
- Batch processing capabilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd water-body-segmentation

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Dataset Structure

Organize your dataset as follows:
```
Water Bodies Dataset/
├── Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Masks/
    ├── mask1.png
    ├── mask2.png
    └── ...
```

## Usage

### 1. Data Preparation
```bash
python3 prepare_data.py
```

### 2. Training
```bash
python3 train.py
```

### 3. Evaluation and Inference
```bash
python3 evaluate.py
```

## Model Architecture

The implementation uses a standard U-Net architecture with:
- Encoder: 4 downsampling blocks with double convolutions
- Bottleneck: 1024 feature channels
- Decoder: 4 upsampling blocks with skip connections
- Output: Single channel for binary segmentation

## Training Configuration

Key hyperparameters:
- Learning rate: 1e-4
- Batch size: 16
- Input size: 256x256
- Loss function: Combined BCE + Dice Loss
- Optimizer: Adam with weight decay

## Evaluation Metrics

- Dice Score (F1)
- Intersection over Union (IoU)
- Pixel Accuracy
- Precision/Recall
- Sensitivity/Specificity

## Results
### 1. Baseline U-Net architecture:
Model performance on test set:

IoU Score:

  Mean: 0.7192 ± 0.2086
  
  Range: [0.0581, 1.0000]

  
Overall Dataset Metrics:

  Accuracy: 0.9081
  
  Precision: 0.8601
  
  Recall: 0.8582
  
  F1-Score: 0.8591


### 2. Residual blocks and Dropout layers to the encoder and bottleneck.
Model performance on test set:

IoU Score:

  Mean: 0.7622 ± 0.2079
  
  Range: [0.0321, 1.0000]

Overall Dataset Metrics:

  Accuracy: 0.9286
  
  Precision: 0.9119
  
  Recall: 0.8650
  
  F1-Score: 0.8878


### 3. Integrate attention modules into skip connections.

IoU Score:

  Mean: 0.7530 ± 0.2232
  
  Range: [0.0227, 1.0000]


Overall Dataset Metrics:

  Accuracy: 0.9212
  
  Precision: 0.8856
  
  Recall: 0.8712
  
  F1-Score: 0.8783

