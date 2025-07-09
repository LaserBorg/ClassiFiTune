import json
import yaml


def load_config(config_path):
    """Load config from a YAML or JSON file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError('Unsupported config file format. Use .yaml, .yml, or .json')
