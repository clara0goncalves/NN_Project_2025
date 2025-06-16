import os
import subprocess
import sys
import shutil
import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CHANGED: Import the data loading function ---
from utils.prepare_data import get_data_loaders

def display_menu():
    """Displays the main interactive menu to the user."""
    # ... (unchanged) ...
    print("\n" + "="*35); print("--- Water Body Segmentation ---"); print("="*35)
    print("1. Setup Environment"); print("2. Split Raw Dataset (Run once)")
    print("3. Train New Model"); print("4. Evaluate Best Model"); print("5. Exit")
    print("="*35)

def setup_environment():
    """Installs dependencies from requirements.txt."""
    # ... (unchanged) ...
    print("\n--- Setting up the environment ---"); requirements_path = 'requirements.txt'
    if not os.path.exists(requirements_path): print(f"Error: '{requirements_path}' not found!"); return
    print("Installing required packages...");
    try: subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path]); print("Setup complete.")
    except subprocess.CalledProcessError as e: print(f"An error occurred: {e}")

def split_dataset():
    """Splits the raw dataset into train/val/test sets."""

    print("\n--- Starting Dataset Split ---"); SOURCE_DATA_DIR = 'Water-Bodies-Dataset/'; DEST_DATA_DIR = 'datasets/'
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15; RANDOM_STATE = 42
    if not os.path.exists(SOURCE_DATA_DIR): print(f"Error: Source '{SOURCE_DATA_DIR}' not found."); return
    if os.path.exists(DEST_DATA_DIR):
        if input(f"Warning: '{DEST_DATA_DIR}' exists. Overwrite? (y/n): ").lower() != 'y': print("Split cancelled."); return
        shutil.rmtree(DEST_DATA_DIR)
    image_paths = sorted(glob.glob(os.path.join(SOURCE_DATA_DIR, 'Images', '*.*')))
    mask_paths = sorted(glob.glob(os.path.join(SOURCE_DATA_DIR, 'Masks', '*.*')))
    if not image_paths or len(image_paths) != len(mask_paths): print("Error: File mismatch or not found."); return
    print(f"Found {len(image_paths)} image/mask pairs.")
    for split in ['train', 'val', 'test']: os.makedirs(os.path.join(DEST_DATA_DIR, split, 'images'), exist_ok=True); os.makedirs(os.path.join(DEST_DATA_DIR, split, 'masks'), exist_ok=True)
    train_imgs, temp_imgs, train_msks, temp_msks = train_test_split(image_paths, mask_paths, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_STATE)
    val_imgs, test_imgs, val_msks, test_msks = train_test_split(temp_imgs, temp_msks, test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), random_state=RANDOM_STATE)
    datasets = {'train': (train_imgs, train_msks), 'val': (val_imgs, val_msks), 'test': (test_imgs, test_msks)}
    for split, (images, masks) in datasets.items():
        print(f"\nCopying {split} files..."); dest_img = os.path.join(DEST_DATA_DIR, split, 'images'); dest_mask = os.path.join(DEST_DATA_DIR, split, 'masks')
        for f in tqdm(images, desc=f"Images to {split}"): shutil.copy(f, dest_img)
        for f in tqdm(masks, desc=f"Masks to {split}"): shutil.copy(f, dest_mask)
    print("\n--- Dataset split complete! ---")

def train_model():
    """Wrapper to initiate the model training process."""
    # ... (unchanged) ...
    print("\n--- Initializing Training ---")
    try: from model.train import main as train_main; train_main()
    except Exception as e: print(f"An error occurred during training: {e}")

def evaluate_model():
    """Interactively finds and evaluates a trained model."""
    print("\n--- Evaluating a Trained Model ---")
    checkpoints_dir = 'checkpoints'
    try:
        experiments = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
        if not experiments: print("Error: No trained models found in 'checkpoints'."); return
        print("Please choose a model to evaluate:"); [print(f"  {i + 1}: {exp}") for i, exp in enumerate(experiments)]
        choice_idx = int(input(f"Enter your choice (1-{len(experiments)}): ")) - 1
        if not 0 <= choice_idx < len(experiments): print("Invalid choice."); return
        
        chosen_experiment = experiments[choice_idx]
        model_path = os.path.join(checkpoints_dir, chosen_experiment, 'best.pth')
        if not os.path.exists(model_path): print(f"Error: 'best.pth' not found in '{chosen_experiment}'."); return
            
        print(f"\nLoading model and config from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint.get('config')
        if not config: sys.exit("Error: Config not found in checkpoint.")

        # --- CHANGED: Get the test loader here ---
        print("Preparing test data loader...")
        _, _, test_loader = get_data_loaders(
            batch_size=config.get('batch_size', 16),
            num_workers=config.get('num_workers', 4)
        )
        if test_loader is None: return # Exit if data loading failed

        from model.eval import ModelEvaluator
        evaluator = ModelEvaluator(model_path, config)
        evaluator.evaluate_dataset(test_loader)

    except (ValueError, IndexError): print("Invalid input.")
    except Exception as e: print(f"An error occurred during evaluation: {e}")

def main():
    """Main function to run the interactive script."""
    # ... (unchanged) ...
    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ")
        if choice == '1': setup_environment()
        elif choice == '2': split_dataset()
        elif choice == '3': train_model()
        elif choice == '4': evaluate_model()
        elif choice == '5': print("Exiting. Goodbye!"); break
        else: print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()