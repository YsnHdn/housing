from pathlib import Path
import yaml


def load_config():
    config_path = Path('config.yaml')
    
    with open(config_path , 'r') as file :
        config = yaml.safe_load(file)
        
    return config

CONFIG = load_config()
    