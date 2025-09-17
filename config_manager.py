"""
Production Configuration Management
==================================

Centralized configuration with environment variable support and validation.
Supports multiple deployment environments (development, staging, production).

Example usage:
    # Use default config
    config = ConfigManager.get_config()
    
    # Load specific environment
    config = ConfigManager.get_config("production")
    
    # Override specific settings
    config = ConfigManager.get_config(overrides={"anomaly_threshold": 3.0})
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    catboost_iterations: int = 500
    catboost_depth: int = 8
    catboost_learning_rate: float = 0.1
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "catboost": 0.7,
        "lstm": 0.3
    })
    

@dataclass
class AnomalyConfig:
    """Anomaly detection configuration."""
    anomaly_threshold: float = 2.5
    forecast_threshold: float = 2.0
    peer_threshold: float = 2.5
    share_absolute: float = 0.40
    share_increase: float = 0.15
    

@dataclass
class SystemConfig:
    """System-level configuration."""
    simulation_years: int = 5
    base_interval_minutes: int = 15
    regions: list = field(default_factory=lambda: [
        "region_1", "region_2", "region_3", "region_4",
        "region_5", "region_6", "region_7", "region_8"
    ])
    gpu_enabled: bool = True
    log_level: str = "INFO"
    

@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = 8501
    auto_refresh_seconds: int = 30
    theme: str = "dark"
    

@dataclass
class DataConfig:
    """Data handling configuration."""
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    cache_enabled: bool = True
    

@dataclass
class ProductionConfig:
    """Complete production configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {filepath}")


class ConfigManager:
    """Configuration manager with environment support."""
    
    _configs: Dict[str, ProductionConfig] = {}
    
    @classmethod
    def get_config(
        self, 
        environment: str = "default",
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ProductionConfig:
        """
        Get configuration for specified environment.
        
        Args:
            environment: Environment name (default, development, staging, production)
            config_file: Path to custom config file
            overrides: Dictionary of config overrides
            
        Returns:
            ProductionConfig instance
        """
        if environment not in self._configs:
            self._configs[environment] = self._load_config(
                environment, config_file, overrides
            )
        
        return self._configs[environment]
    
    @classmethod
    def _load_config(
        self,
        environment: str,
        config_file: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ProductionConfig:
        """Load configuration from file and environment variables."""
        
        # Start with default config
        config = ProductionConfig()
        
        # Load from file if specified
        if config_file:
            config = self._load_from_file(config_file)
        elif environment != "default":
            # Look for environment-specific config file
            config_path = Path(f"config/{environment}.json")
            if config_path.exists():
                config = self._load_from_file(config_path)
        
        # Apply environment variables
        config = self._apply_env_vars(config)
        
        # Apply overrides
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        logger.info(f"Loaded configuration for environment: {environment}")
        return config
    
    @classmethod
    def _load_from_file(self, filepath: Union[str, Path]) -> ProductionConfig:
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filepath}, using defaults")
            return ProductionConfig()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return ProductionConfig(
                model=ModelConfig(**data.get("model", {})),
                anomaly=AnomalyConfig(**data.get("anomaly", {})),
                system=SystemConfig(**data.get("system", {})),
                dashboard=DashboardConfig(**data.get("dashboard", {})),
                data=DataConfig(**data.get("data", {}))
            )
        
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
            return ProductionConfig()
    
    @classmethod
    def _apply_env_vars(self, config: ProductionConfig) -> ProductionConfig:
        """Apply environment variable overrides."""
        
        # System config from env vars
        config.system.simulation_years = int(os.getenv(
            "SMARTGRID_SIMULATION_YEARS", config.system.simulation_years
        ))
        config.system.gpu_enabled = os.getenv(
            "SMARTGRID_GPU_ENABLED", str(config.system.gpu_enabled)
        ).lower() == "true"
        config.system.log_level = os.getenv(
            "SMARTGRID_LOG_LEVEL", config.system.log_level
        )
        
        # Anomaly detection from env vars
        config.anomaly.anomaly_threshold = float(os.getenv(
            "SMARTGRID_ANOMALY_THRESHOLD", config.anomaly.anomaly_threshold
        ))
        config.anomaly.forecast_threshold = float(os.getenv(
            "SMARTGRID_FORECAST_THRESHOLD", config.anomaly.forecast_threshold
        ))
        
        # Dashboard from env vars
        config.dashboard.host = os.getenv(
            "SMARTGRID_DASHBOARD_HOST", config.dashboard.host
        )
        config.dashboard.port = int(os.getenv(
            "SMARTGRID_DASHBOARD_PORT", config.dashboard.port
        ))
        
        # Data directories from env vars
        config.data.data_dir = os.getenv(
            "SMARTGRID_DATA_DIR", config.data.data_dir
        )
        config.data.models_dir = os.getenv(
            "SMARTGRID_MODELS_DIR", config.data.models_dir
        )
        config.data.outputs_dir = os.getenv(
            "SMARTGRID_OUTPUTS_DIR", config.data.outputs_dir
        )
        
        return config
    
    @classmethod
    def _apply_overrides(
        self, 
        config: ProductionConfig, 
        overrides: Dict[str, Any]
    ) -> ProductionConfig:
        """Apply configuration overrides."""
        
        for key, value in overrides.items():
            # Handle nested config updates
            if "." in key:
                section, setting = key.split(".", 1)
                if hasattr(config, section):
                    section_config = getattr(config, section)
                    if hasattr(section_config, setting):
                        setattr(section_config, setting, value)
            else:
                # Handle direct anomaly threshold overrides (for backward compatibility)
                if hasattr(config.anomaly, key):
                    setattr(config.anomaly, key, value)
                elif hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        return config
    
    @classmethod
    def create_sample_configs(self) -> None:
        """Create sample configuration files for different environments."""
        
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Development config
        dev_config = ProductionConfig()
        dev_config.system.simulation_years = 1  # Faster for development
        dev_config.system.log_level = "DEBUG"
        dev_config.dashboard.auto_refresh_seconds = 10
        dev_config.save(config_dir / "development.json")
        
        # Staging config
        staging_config = ProductionConfig()
        staging_config.system.simulation_years = 3
        staging_config.anomaly.anomaly_threshold = 2.0  # More sensitive
        staging_config.dashboard.host = "0.0.0.0"  # Allow external access
        staging_config.save(config_dir / "staging.json")
        
        # Production config
        prod_config = ProductionConfig()
        prod_config.system.log_level = "WARNING"
        prod_config.anomaly.anomaly_threshold = 3.0  # Less false positives
        prod_config.dashboard.host = "0.0.0.0"
        prod_config.data.cache_enabled = True
        prod_config.save(config_dir / "production.json")
        
        logger.info("Sample configuration files created in config/ directory")


# Convenience function for backward compatibility
def get_grid_config(overrides: Optional[Dict[str, Any]] = None) -> ProductionConfig:
    """Get grid configuration with optional overrides."""
    return ConfigManager.get_config(overrides=overrides)


if __name__ == "__main__":
    # Create sample configurations
    ConfigManager.create_sample_configs()
    
    # Example usage
    config = ConfigManager.get_config("development")
    print(f"Development config: {config.system.simulation_years} years")
    
    config = ConfigManager.get_config(overrides={"anomaly_threshold": 4.0})
    print(f"Override config: {config.anomaly.anomaly_threshold} threshold")