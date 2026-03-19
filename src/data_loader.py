import os
import torch
from pydub import AudioSegment

DRONE_TIMEFRAMES = {
    "mavic1": {"start": (3 * 60 + 5) * 1000,   # 03:05
               "end": (20 * 60 + 26) * 1000},  # 20:26
               
    "mavic2": {"start": (24 * 60 + 51) * 1000, # 24:51
               "end": (37 * 60 + 56) * 1000},  # 37:56
               
    "mini":   {"start": (41 * 60 + 48) * 1000, # 41:48
               "end": (53 * 60 + 10) * 1000}   # 53:10
}

def extract_drone_classes(input_file, base_output_dir, chunk_length_ms=100):
    """Slices specific timeframes from the audio and sorts them by drone type."""
    print(f"Loading large audio file: {input_file}...")
    audio = AudioSegment.from_wav(input_file).set_channels(1)
    
    for drone_name, times in DRONE_TIMEFRAMES.items():
        print(f"\nProcessing {drone_name}...")
        
        # Create a specific folder for this drone
        output_dir = os.path.join(base_output_dir, drone_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Isolate the segment where this specific drone is flying
        drone_segment = audio[times["start"]:times["end"]]
        total_chunks = len(drone_segment) // chunk_length_ms
        
        print(f"Slicing into {total_chunks} chunks...")
        
        for i in range(total_chunks):
            start_time = i * chunk_length_ms
            end_time = start_time + chunk_length_ms
            chunk = drone_segment[start_time:end_time]
            
            samples = chunk.get_array_of_samples()
            tensor_data = torch.tensor(samples, dtype=torch.float32)
            
            if tensor_data.max() > 0:
                tensor_data = tensor_data / tensor_data.abs().max()
                
            torch.save(tensor_data, os.path.join(output_dir, f"{drone_name}_{i:05d}.pt"))

if __name__ == '__main__':
    INPUT_FILE = "data/raw/Bruel 4006 - Bal elso_01.wav"
    OUTPUT_FOLDER = "data/processed/"
    
    extract_drone_classes(INPUT_FILE, OUTPUT_FOLDER)
    print("\nData extraction complete!")