import subprocess
import shlex
import os

def run_command(command):
    """
    Executes a command in the shell and allows it to print directly to the console.
    """
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        if os.name == 'nt' and command.startswith('python3'):
            command = 'python' + command[len('python3'):]
        print(f"Executing command: {command}")
        subprocess.run(
            shlex.split(command), 
            check=True,
            cwd=project_root
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def prepare_data():
    """Runs the data preparation script from its new location in utils/."""
    print("\n--- Preparing Data ---")
    command = "python3 utils/prepare_data.py"
    run_command(command)
    print("\n--- Data Preparation Finished ---")

def visualize_dataset():
    """Runs the dataset visualization script from its new location in utils/."""
    print("\n--- Visualizing Dataset ---")
    command = "python3 utils/visualize_dataset.py"
    run_command(command)
    print("\n--- Dataset Visualization Finished ---")

def train_model():
    """
    Guides the user through the process of training a model with streamlined defaults.
    """
    print("\n--- Train a New Model ---")

    print("Select a model to train:")
    print("1. AER U-Net")
    print("2. U-Net++ (Pre-trained Encoder)")
    print("3. SegFormer-B4")
    model_choice = input("Enter your choice (1-3): ")

    model_map = {
        "1": "aer-unet",
        "2": "unet++-pretrained-encoder",
        "3": "segformer-b4",
    }
    model = model_map.get(model_choice)

    if not model:
        print("Invalid choice. Please try again.")
        return

    print("\nEnter training parameters (press Enter to use default values):")
    num_epochs = input("Number of epochs (default: 100): ") or "100"
    
    default_lr = "5e-5" if model == "segformer-b4" else "1e-4"
    learning_rate = input(f"Learning rate (default: {default_lr}): ") or default_lr
    
    batch_size = input("Batch size (default: 16): ") or "16"

    loss_options = "bce, dice, combined, focal, tversky, focal_lovasz"
    loss_type = input(f"Loss type ({loss_options}; default: focal_lovasz): ") or "focal_lovasz"
    
    command = f"python3 train.py --model {model} --num_epochs {num_epochs} --learning_rate {learning_rate} --batch_size {batch_size} --loss_type {loss_type}"

    if loss_type == 'focal_lovasz':
        print("\n--- Configure FocalLovaszLoss Weights ---")
        focal_weight = input("Focal loss weight (default: 0.5): ") or "0.5"
        lovasz_weight = input("Lovasz loss weight (default: 0.5): ") or "0.5"
        command += f" --focal_weight {focal_weight} --lovasz_weight {lovasz_weight}"

    print("\n--- Configure Training Options (default is Yes) ---")

    # FIX: Ask the user for each option, with 'Yes' as the default.
    # The feature is enabled unless the user explicitly types 'n'.
    if input("Use Automatic Mixed Precision (AMP)? (Y/n): ").lower() != 'n':
        command += " --use_amp"

    if model == "unet++-pretrained-encoder":
        if input("Use Deep Supervision? (Y/n): ").lower() != 'n':
            command += " --deep_supervision"
    
    if input("Use Early Stopping? (Y/n): ").lower() != 'n':
        command += " --early_stopping"
    
    if input("Use gradient clipping? (y/N): ").lower() == 'y':
        command += " --gradient_clipping"

    print(f"\nFinal command to be executed:\n{command}\n")
    run_command(command)
    print("\n--- Model Training Finished ---")

def evaluate_model():
    """Launches the fully interactive evaluation script."""
    print("\n--- Launching Interactive Evaluation Script ---")
    command = "python3 evaluate.py"
    run_command(command)
    print("\n--- Evaluation Finished ---")

def install_requirements():
    """Installs or updates the required Python packages from requirements.txt."""
    print("\n--- Installing/Updating Requirements ---")
    command = "python3 -m pip install -r requirements.txt"
    run_command(command)
    print("\n--- Requirements Installation Finished ---")

def main_menu():
    """Displays the main interactive menu to the user."""
    while True:
        print("\n============================")
        print("  Water Segmentation Menu")
        print("============================")
        print("1. Install/Update Requirements")
        print("2. Prepare Data (Split Raw Dataset)")
        print("3. Visualize a Batch of the Dataset")
        print("4. Train a Model")
        print("5. Evaluate a Model")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ")

        if choice == "1":
            install_requirements()
        elif choice == "2":
            prepare_data() 
        elif choice == "3":
            visualize_dataset()
        elif choice == "4":
            train_model()
        elif choice == "5":
            evaluate_model()
        elif choice == "6":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main_menu()