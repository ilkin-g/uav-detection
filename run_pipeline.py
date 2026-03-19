import os
import torch
from src.data_loader import extract_drone_classes
from src.train import train_model

def main():
    print("=== UAV Multi-Class Detection Pipeline using RGW-VP ===")
    
    raw_audio_file = "data/raw/Bruel 4006 - Bal elso_01.wav"
    processed_dir = "data/processed/"
    model_save_dir = "models/"
    
    if not os.path.exists(os.path.join(processed_dir, "mavic1")):
        print("\n--- Step 1: Extracting Drone Classes from Raw Audio ---")
        if os.path.exists(raw_audio_file):
            extract_drone_classes(raw_audio_file, processed_dir, chunk_length_ms=100)
        else:
            print(f"Error: Could not find '{raw_audio_file}'.")
            print("Please ensure your audio file is placed in 'data/raw/'.")
            return
    else:
        print(f"\n--- Step 1: Data already categorized in '{processed_dir}'. Skipping extraction. ---")

    print("\n--- Step 2: Training the Multi-Class RGW-VP Neural Network ---")
    trained_model = train_model(
        data_dir=processed_dir, 
        epochs=15,
        batch_size=64,
        learning_rate=0.001
    )
    
    if trained_model:
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, "uav_multiclass_weights.pth")
        
        torch.save(trained_model.state_dict(), save_path)
        print(f"\n=== Pipeline Execution Successful! ===")
        print(f"Model weights safely stored in: {save_path}")

if __name__ == "__main__":
    main()