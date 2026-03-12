import os
from pydub import AudioSegment
import torch

def chunk_large_audio(input_file, output_dir, chunk_length_ms=100):
    """Slices a massive audio file into 100ms segments and saves them as PyTorch tensors."""
    print(f"Loading {input_file}...")
    
    audio = AudioSegment.from_wav(input_file)
    
    audio = audio.set_channels(1) 
    
    total_chunks = len(audio) // chunk_length_ms
    print(f"Total 100ms chunks to process: {total_chunks}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(total_chunks):
        start_time = i * chunk_length_ms
        end_time = start_time + chunk_length_ms
        chunk = audio[start_time:end_time]
        
        # Get raw data and convert to PyTorch tensor
        samples = chunk.get_array_of_samples()
        tensor_data = torch.tensor(samples, dtype=torch.float32)
        
        # Normalize the audio segment
        if tensor_data.max() > 0:
            tensor_data = tensor_data / tensor_data.abs().max()
            
        # Save chunk to disk
        torch.save(tensor_data, os.path.join(output_dir, f"chunk_{i}.pt"))

if __name__ == '__main__':
    INPUT_FILE = "./data/raw/Bruel 4006 - Bal elso_01.wav"
    OUTPUT_FOLDER = "./data/processed/"
    
    chunk_large_audio(INPUT_FILE, OUTPUT_FOLDER)
    print("Chunking complete!")