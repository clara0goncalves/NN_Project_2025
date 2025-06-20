import subprocess
import shlex
import os

def run_command(command):
    """
    Executes a command in the shell and allows it to print directly to the console.
    This function is used to run the different scripts of the project.
    """
    try:
        subprocess.run(
            shlex.split(command), 
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

    except FileNotFoundError:
        print(f"Error: The command '{command.split()[0]}' was not found.")
        print("Please ensure that the script is in the same directory or in the system's PATH.")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Command returned non-zero exit code {e.returncode}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def prepare_data():
    """
    Runs the data preparation script as a module to handle imports correctly.
    """
    print("\n--- Preparing Data ---")
    command = "python3 -m src.preprocessing.prepare_data"
    run_command(command)
    print("\n--- Data Preparation Finished ---")

def visualize_dataset():
    """
    Runs the dataset visualization script as a module.
    """
    print("\n--- Visualizing Dataset ---")
    command = "python3 -m src.preprocessing.visualize_dataset"
    run_command(command)
    print("\n--- Dataset Visualization Finished ---")

def train_model():
    """
    Guides the user through the process of training a model.
    """
    print("\n--- Train a New Model ---")

    print("Select a model to train:")
    print("1. Basic U-Net")
    print("2. Enhanced U-Net")
    print("3. Attention U-Net")
    print("4. U-Net++")
    print("5. AER U-Net")
    print("6. U-Net++ with pre-trained encoder")
    model_choice = input("Enter your choice (1-6): ")

    model_map = {
        "1": "unet", 
        "2": "enhanced", 
        "3": "attention", 
        "4": "unet++", 
        "5": "aer-unet",
        "6": "unet++-pretrained-encoder"
    }
    model = model_map.get(model_choice)

    if not model:
        print("Invalid choice. Please try again.")
        return

    print("\nEnter training parameters (press Enter to use default values):")
    num_epochs = input("Number of epochs (default: 50): ") or "50"
    learning_rate = input("Learning rate (default: 1e-4): ") or "1e-4"
    batch_size = input("Batch size (default: 16): ") or "16"
    loss_type = input("Loss type (e.g., combined, lovasz, focal_lovasz; default: combined): ") or "combined"
    command = f"python3 train.py --model {model} --num_epochs {num_epochs} --learning_rate {learning_rate} --batch_size {batch_size} --loss_type {loss_type}"

    if loss_type == 'focal_lovasz':
        print("\n--- Configure FocalLovaszLoss Weights ---")
        focal_weight = input("Focal loss weight (default: 0.5): ") or "0.5"
        lovasz_weight = input("Lovasz loss weight (default: 0.5): ") or "0.5"
        command += f" --focal_weight {focal_weight} --lovasz_weight {lovasz_weight}"

    if model in ["enhanced", "attention", "unet++", "aer-unet"]:
        if input("Use mixed precision (AMP)? (Y/n): ").lower() != 'n':
            command += " --use_amp"
        
        if model == 'unet++':
            if input("Use deep supervision? (Y/n): ").lower() != 'n':
                command += " --deep_supervision"

    if input("Use early stopping? (Y/n): ").lower() != 'n':
        command += " --early_stopping"
    
    if input("Use gradient clipping? (Y/n): ").lower() != 'n':
        command += " --gradient_clipping"

    print(f"\nExecuting command: {command}\n")
    run_command(command)
    print("\n--- Model Training Finished ---")

# --- MODIFIED: The evaluate_model function is now interactive ---
def evaluate_model():
    """
    Launches the interactive evaluation script, asking the user if they
    want to use CRF post-processing.
    """
    print("\n--- Launching Interactive Evaluation ---")
    
    # The evaluation script itself is interactive for model selection.
    # We add an interactive prompt here for the CRF option.
    use_crf_choice = input("Use CRF post-processing to refine masks? (Y/n): ").lower()
    
    if use_crf_choice != 'n':
        # If user inputs 'y', 'yes', or just presses Enter
        command = "python3 evaluate.py --use_crf"
        print("CRF enabled.")
    else:
        # If user inputs 'n' or 'no'
        command = "python3 evaluate.py"
        print("CRF disabled.")

    print(f"\nExecuting command: {command}\n")
    run_command(command)
    print("\n--- Evaluation Finished ---")

def install_requirements():
    """
    Installs or updates the required Python packages from requirements.txt.
    """
    print("\n--- Installing/Updating Requirements ---")
    command = "python3 -m pip install -r requirements.txt"
    run_command(command)
    print("\n--- Requirements installation finished ---")

def main_menu():
    """
    Displays the main menu and handles user input.
    """
    while True:
        print("\n=====================")
        print("  Interactive Menu")
        print("=====================")
        print("1. Prepare Data")
        print("2. Visualize Dataset")
        print("3. Train a New Model")
        print("4. Evaluate a Trained Model")
        print("5. Install/Update Requirements")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")

        if choice == "1":
            prepare_data()
        elif choice == "2":
            visualize_dataset()
        elif choice == "3":
            train_model()
        elif choice == "4":
            evaluate_model()
        elif choice == "5":
            install_requirements()
        elif choice == "6":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main_menu()