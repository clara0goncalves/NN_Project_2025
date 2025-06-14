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