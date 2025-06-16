# main.py
import os
import subprocess
import sys
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch # ADDED for loading config

# Add project root to path to allow imports from model, utils, etc.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def display_menu():
    # ... (unchanged)
    print("\n--- Water Body Segmentation ---")
    print("Please choose an option:")
    print("1. Setup Environment")
    print("2. Split Raw Dataset (Run this once!)")
    print("3. Train Model")
    print("4. Evaluate Model")
    print("5. Exit")
    print("---------------------------------")

def setup_environment():
    # ... (unchanged)
    print("\n--- Setting up the environment ---")
    requirements_path = 'requirements.txt'
    if not os.path.exists(requirements_path):
        print(f"Error: {requirements_path} not found!")
        return
    print("Installing required packages from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
        print("Environment setup complete.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during setup: {e}")

def split_dataset():
    # ... (unchanged)
    print("\n--- Starting Dataset Split ---")
    SOURCE_DATA_DIR = 'Water-Bodies-Dataset/'
    DEST_DATA_DIR = 'datasets/'
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15
    RANDOM_STATE = 42
    if not os.path.exists(SOURCE_DATA_DIR):
        print(f"Error: Source directory '{SOURCE_DATA_DIR}' not found.")
        print("Please make sure your raw data is in that folder.")
        return
    if os.path.exists(DEST_DATA_DIR):
        overwrite = input(f"Warning: Destination '{DEST_DATA_DIR}' already exists. Overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("Split operation cancelled.")
            return
        shutil.rmtree(DEST_DATA_DIR)
        print("Removed existing destination directory.")
    source_images_dir = os.path.join(SOURCE_DATA_DIR, 'Images')
    source_masks_dir = os.path.join(SOURCE_DATA_DIR, 'Masks')
    image_paths = sorted(glob.glob(os.path.join(source_images_dir, '*.jpg')))
    mask_paths = sorted(glob.glob(os.path.join(source_masks_dir, '*.jpg')))
    if not image_paths or not mask_paths or len(image_paths) != len(mask_paths):
        print("Error: Image/mask mismatch or files not found in source directory.")
        return
    print(f"Found {len(image_paths)} image/mask pairs.")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(DEST_DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_DATA_DIR, split, 'masks'), exist_ok=True)
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(
        image_paths, mask_paths, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_STATE
    )
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(
        temp_imgs, temp_msks, test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), random_state=RANDOM_STATE
    )
    datasets = {
        'train': (train_imgs, train_msks),
        'val': (val_imgs, val_msks),
        'test': (test_imgs, test_msks)
    }
    for split, (images, masks) in datasets.items():
        print(f"\nCopying {split} files...")
        dest_img_dir = os.path.join(DEST_DATA_DIR, split, 'images')
        dest_msk_dir = os.path.join(DEST_DATA_DIR, split, 'masks')
        for f in tqdm(images, desc=f"Images to {split}"):
            shutil.copy(f, dest_img_dir)
        for f in tqdm(masks, desc=f"Masks to {split}"):
            shutil.copy(f, dest_msk_dir)
    print("\n--- Dataset split complete! ---")

def train_model():
    # ... (unchanged)
    print("\n--- Training the model ---")
    try:
        from model.train import main as train_main
        train_main()
        print("Training finished.")
    except ImportError:
        print("Error: Could not import training module. Make sure all files are in place.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

## CHANGED: This function is now interactive and automatically loads the config
def evaluate_model():
    """Interactively select a model and evaluate it."""
    print("\n--- Evaluating a Trained Model ---")
    checkpoints_dir = 'checkpoints'
    
    try:
        # Find all available experiment directories
        experiments = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        if not experiments:
            print("Error: No trained models found in the 'checkpoints' directory.")
            return

        # Let the user choose which experiment to evaluate
        print("Please choose a model to evaluate:")
        for i, exp_name in enumerate(experiments):
            print(f"{i + 1}: {exp_name}")
        
        choice = input(f"Enter your choice (1-{len(experiments)}): ")
        choice_idx = int(choice) - 1
        
        if not 0 <= choice_idx < len(experiments):
            print("Invalid choice.")
            return

        chosen_experiment = experiments[choice_idx]
        model_path = os.path.join(checkpoints_dir, chosen_experiment, 'best.pth')

        if not os.path.exists(model_path):
            print(f"Error: 'best.pth' not found in '{chosen_experiment}'.")
            return
            
        # Automatically load the configuration from the checkpoint
        print(f"Loading model and config from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # The config might not be in older checkpoints, so handle that case
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Fallback for older models that didn't save config in the checkpoint
            print("Warning: Config not found in checkpoint. Using a default config.")
            config = {'n_channels': 3, 'n_classes': 1}

        from model.eval import ModelEvaluator
        from utils.prepare_data import test_loader
        
        evaluator = ModelEvaluator(model_path, config)
        evaluator.evaluate_dataset(test_loader)
        print("Evaluation finished.")

    except (ValueError, IndexError):
        print("Invalid input. Please enter a number from the list.")
    except ImportError:
        print("Error: Could not import evaluation modules. Make sure all files are in place.")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

# main loop is unchanged
def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            setup_environment()
        elif choice == '2':
            split_dataset()
        elif choice == '3':
            train_model()
        elif choice == '4':
            evaluate_model()
        elif choice == '5':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()