import yaml
from pathlib import Path

def load_config(config_path: str):
    """
    Loads a YAML configuration file.
    
    Args:
        config_path (str): Path to validity YAML file.
        
    Returns:
        dict: Parsed configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
