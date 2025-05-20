import os
import numpy as np
from scapy.all import rdpcap
from tqdm import tqdm

# Paths
PCAP_FOLDER = "../pcaps"
SAVE_FOLDER = "./dataset/images"

# Ensure output folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

def pcap_to_image(packet_data, width=28, height=28):
    # Convert raw bytes to an image-like 2D numpy array
    byte_data = bytes(packet_data)
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    # Pad or truncate to fixed size
    target_size = width * height
    if len(byte_array) < target_size:
        byte_array = np.pad(byte_array, (0, target_size - len(byte_array)), 'constant')
    else:
        byte_array = byte_array[:target_size]

    return byte_array.reshape((width, height))

def preprocess_pcap(file_path, output_path, label):
    try:
        packets = rdpcap(file_path)
        for idx, packet in enumerate(packets):
            image = pcap_to_image(bytes(packet))
            save_path = os.path.join(output_path, f"{label}_{idx}.npy")
            np.save(save_path, image)
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")

def run_preprocessing():
    print("🚀 Starting preprocessing...")
    files = [f for f in os.listdir(PCAP_FOLDER) if f.endswith('.pcap')]

    for i, file in enumerate(tqdm(files, desc="Processing PCAPs")):
        file_path = os.path.join(PCAP_FOLDER, file)
        label = os.path.splitext(file)[0]  # e.g., "Facetime" from "Facetime.pcap"
        preprocess_pcap(file_path, SAVE_FOLDER, label)

    print("✅ Preprocessing complete!")

if __name__ == "__main__":
    run_preprocessing()
