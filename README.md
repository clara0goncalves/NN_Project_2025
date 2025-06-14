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

### 3. Evaluation
```bash
python3 evaluate.py
```

### 4. Inference
```python
from utils.inference_utils import WaterBodySegmentor

# Initialize segmentor
segmentor = WaterBodySegmentor('checkpoints/best.pth')

# Single image prediction
segmentor.visualize_prediction('test_image.jpg')

# Batch processing
segmentor.batch_predict('input_folder', 'output_folder')
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

Model performance on test set:
- Dice Score: XX.XX ± XX.XX
- IoU Score: XX.XX ± XX.XX
- Pixel Accuracy: XX.XX%

## Citation

If you use this code, please cite:
```
@article{your_paper,
  title={Water Body Segmentation using U-Net},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
"""

# Installation script (install.sh)
"""
#!/bin/bash

echo "Setting up Water Body Segmentation project..."

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p checkpoints runs evaluation_results inference_results

echo "Setup completed successfully!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To start training, run: python train.py"
