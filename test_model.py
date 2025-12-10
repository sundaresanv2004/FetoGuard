import torch
from src.models.unet import UNet

def test_model():
    print("Initializing U-Net...")
    model = UNet(n_channels=3, n_classes=1)
    
    # Create dummy input: Batch Size=4, Channels=3, Height=256, Width=256
    batch_size = 4
    channels = 3
    height, width = 256, 256
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    print("Running forward pass...")
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Check output shape
    assert logits.shape == (batch_size, 1, height, width), "Output shape incorrect!"
    
    print("✅ Model verification successful!")

if __name__ == "__main__":
    test_model()
