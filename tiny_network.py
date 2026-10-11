import torch
import torch.nn as nn

# 1. Define the AI Model as a Python Class
# Every PyTorch model MUST inherit from "nn.Module" (Neural Network Module)
class TinyNetwork(nn.Module):
    def __init__(self):
        super(TinyNetwork, self).__init__() # Required setup step
        
        # Here we define the "parts" of our model (like Lego blocks).
        # We are using simple Linear layers (also called Fully Connected layers).
        # Later for our images, we will use Convolutional (Conv2d) layers instead.
        self.layer1 = nn.Linear(in_features=10, out_features=5) 
        self.layer2 = nn.Linear(in_features=5, out_features=2)

    # 2. Define the Forward Pass
    # This function is mandatory. It controls how data flows through the parts.
    def forward(self, x):
        print(" -> Passing data through Layer 1...")
        x = self.layer1(x)
        print(" -> Passing data through Layer 2...")
        x = self.layer2(x)
        return x

# --- Running the Code ---
if __name__ == "__main__":
    # 1. "Build" our model (create an instance of the class)
    model = TinyNetwork()
    
    # 2. Create some dummy input data
    # Imagine this is a single piece of data with 10 numbers (features)
    dummy_input = torch.randn(1, 10) 
    
    print("Input Data shape:", dummy_input.shape)
    
    # 3. Pass the data through the model
    # Note: We just call model(dummy_input). We do NOT write model.forward(dummy_input).
    # PyTorch handles calling the forward function behind the scenes automatically.
    output = model(dummy_input)
    
    print("\nOutput Data shape:", output.shape)
    print("Final Output Values:\n", output)
