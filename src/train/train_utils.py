import os
from datetime import datetime

def save_configs(
    model_config: dict,
    dataset_config: dict,
    train_config: dict, 
    output_dir: str, 
    device: str
) -> str:
    """Save all configuration parameters to a text file.
    
    Args:
        model_config: Configuration for the model.
        dataset_config: Configuration for the dataset.
        train_config: Configuration for training parameters.
        output_dir: Directory to save the config file.
        device: Device being used for training.
        
    Returns:
        str: Path to the saved config file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config_path = os.path.join(output_dir, f"experiment_config_{timestamp}.txt")
    
    with open(config_path, 'w') as f:
        f.write("======== EXPERIMENT CONFIGURATION ========\n\n")
        
        f.write("== MODEL CONFIGURATION ==\n")
        for key, value in model_config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n== DATASET CONFIGURATION ==\n")
        for key, value in dataset_config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n== TRAINING CONFIGURATION ==\n")
        for key, value in train_config.items():
            f.write(f"{key}: {value}\n")
            
        f.write("\n== ENVIRONMENT ==\n")
        f.write(f"Device: {device}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Timestamp: {timestamp}\n")
        
    print(f"Configuration saved to {config_path}")
    return config_path
