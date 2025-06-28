import yaml
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional

class ConfigManager:
    """Efficient configuration manager for deep learning models"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file based on extension"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        suffix = self.config_path.suffix.lower()
        
        if suffix == '.yaml' or suffix == '.yml':
            return self._load_yaml()
        elif suffix == '.json':
            return self._load_json()
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    
    def _load_yaml(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_json(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, key: str, value: Any) -> None:
        """Update config value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save current config to file"""
        save_path = Path(path) if path else self.config_path
        
        if save_path.suffix.lower() in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif save_path.suffix.lower() == '.json':
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
