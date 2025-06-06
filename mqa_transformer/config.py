import yaml
from pathlib import Path

def load_config(path: Path, size: str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["models"][size], config["benchmark"]