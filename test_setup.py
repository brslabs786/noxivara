import torch

def check_environment():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if Apple Silicon GPU (Metal Performance Shaders) is available
    if torch.backends.mps.is_available():
        print("GPU Acceleration: ENABLED (Apple MPS detected)")
        
        # Set the device to MPS (the M2 GPU)
        mps_device = torch.device("mps")
        
        # Create a simple mathematical matrix (tensor) and send it to the GPU
        x = torch.ones(3, 3, device=mps_device)
        print("\nSuccess! Created a tensor on the Apple M2 GPU:")
        print(x)
        
    elif torch.backends.mps.is_built():
        print("GPU Acceleration: MPS is built but not available (Check MacOS version).")
    else:
        print("GPU Acceleration: NOT ENABLED. PyTorch will use the CPU.")

if __name__ == "__main__":
    print("--- Stage 1: Environment Check ---")
    check_environment()
