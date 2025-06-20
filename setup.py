# setup.py
from setup tools import setup, find packages

setup(
    name="water-body-segmentation",
    version="1.0.0",
    description="U-Net based water body segmentation using PyTorch",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "opencv-python>=4.6.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0",
        "tensorboard>=2.9.0",
        "Pillow>=9.0.0"
    ],
    python_requires=">=3.8",
)


# Project Structure:
"""
water-body-segmentation/
├── models/
│   ├── __init__.py
│   └── unet.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── metrics.py
│   └── inference_utils.py
├── datasets/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── checkpoints/
├── runs/  # Tensorboard logs
├── evaluation_results/
├── inference_results/
├── prepare_data.py
├── train.py
├── evaluate.py
├── requirements.txt
├── setup.py
└── README.md
"""