import torch.nn as nn
import segmentation_models_pytorch as smp
from src.models.unet import UNet

def get_model(config):
    """
    Returns the model based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing 'model' key.
        
    Returns:
        nn.Module: The requested model.
    """
    model_cfg = config.get('model', {})
    name = model_cfg.get('name', 'unet')
    n_channels = model_cfg.get('in_channels', 3)
    n_classes = model_cfg.get('classes', 1)
    
    if name == 'unet_custom':
        return UNet(n_channels=n_channels, n_classes=n_classes)
        
    elif name == 'unet_pretrained':
        encoder = model_cfg.get('encoder', 'resnet34')
        weights = model_cfg.get('weights', 'imagenet')
        
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=n_channels,
            classes=n_classes,
        )
        return model
        
    else:
        raise ValueError(f"Unknown model name: {name}")
