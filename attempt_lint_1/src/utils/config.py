import yaml
import os

class Config:
    """
    Loads and provides access to hyperparameters and settings from YAML files.
    """
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self._add_attributes(self.config)

    def _load_config(self, config_path):
        """Loads configuration from a YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _add_attributes(self, data, parent_key=''):
        """Recursively adds dictionary keys as attributes."""
        for key, value in data.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                setattr(self, key, Config()) # Create a nested Config object
                self[key]._add_attributes(value, full_key)
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Retrieves a configuration value using dot notation or direct key."""
        parts = key.split('.')
        current = self
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        return current

    def __getitem__(self, key):
        """Allows dictionary-like access."""
        return getattr(self, key)

    def __contains__(self, key):
        """Allows 'in' operator check."""
        return hasattr(self, key)

    def __repr__(self):
        return f"Config({self.config})"

    def __str__(self):
        return yaml.dump(self.config, indent=2)