# enhanced_config.py
"""
Enhanced Configuration for Smart Grid Load Balancing Optimizer
==============================================================
Ensures consistent 15-minute intervals and proper multi-horizon forecasting setup.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class TimeConfig:
    """Time-related configuration."""
    base_interval_minutes: int = 15
    forecast_horizons: Dict[str, int] = field(default_factory=lambda: {
        '15min': 1,      # 1 * 15min = 15 minutes
        '1h': 4,         # 4 * 15min = 1 hour
        '6h': 24,        # 24 * 15min = 6 hours  
        '12h': 48,       # 48 * 15min = 12 hours
        '24h': 24,       # 24 * 1hour = 24 hours (upsampled)
        '48h': 48        # 48 * 1hour = 48 hours (upsampled)
    })
    upsampling_horizons: List[str] = field(default_factory=lambda: ['24h', '48h'])

@dataclass
class ModelConfig:
    """Model training configuration."""
    algorithms: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'catboost'])
    ensemble_method: str = 'weighted_average'
    cross_validation_folds: int = 5
    
    # Direct forecasting strategy configuration
    direct_15min_horizons: List[str] = field(default_factory=lambda: ['15min', '1h', '6h', '12h'])
    direct_hourly_horizons: List[str] = field(default_factory=lambda: ['24h', '48h'])
    
    # Sliding window configuration
    sliding_window_size: int = 96  # 24 hours of 15-min data for context
    sliding_window_step: int = 1   # Step size for sliding window

@dataclass
class EnhancedConfig:
    """Complete enhanced configuration."""
    time: TimeConfig = field(default_factory=TimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Data generation
    simulation_years: int = 5
    train_years: int = 4
    regions: Dict[str, str] = field(default_factory=lambda: {
        'region_1_urban_metro': 'Urban Metro',
        'region_2_suburban_mixed': 'Suburban Mixed',
        'region_3_industrial_heavy': 'Industrial Heavy',
        'region_4_rural_agricultural': 'Rural Agricultural',
        'region_5_coastal_renewable': 'Coastal Renewable',
        'region_6_mountain_hydro': 'Mountain Hydro',
        'region_7_desert_solar': 'Desert Solar',
        'region_8_mixed_commercial': 'Mixed Commercial'
    })
    
    # Feature engineering
    lag_hours: List[int] = field(default_factory=lambda: [1, 2, 4, 12, 24, 48, 96, 168])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 12, 24, 96, 288])
    
    # Deployment
    refresh_interval_minutes: int = 15

# Global enhanced config instance
enhanced_config = EnhancedConfig()

def get_interval_steps(horizon: str) -> int:
    """Get number of 15-minute steps for a given horizon."""
    return enhanced_config.time.forecast_horizons[horizon]

def requires_upsampling(horizon: str) -> bool:
    """Check if horizon requires upsampling from quarters to hours."""
    return horizon in enhanced_config.time.upsampling_horizons

def is_direct_15min_horizon(horizon: str) -> bool:
    """Check if horizon should use direct forecasting with 15-min intervals."""
    return horizon in enhanced_config.model.direct_15min_horizons

def is_direct_hourly_horizon(horizon: str) -> bool:
    """Check if horizon should use direct forecasting with hourly intervals."""
    return horizon in enhanced_config.model.direct_hourly_horizons

def get_horizon_steps(horizon: str) -> int:
    """Get the number of steps for direct forecasting for a horizon."""
    return enhanced_config.time.forecast_horizons[horizon]

def get_sliding_window_config() -> Dict[str, int]:
    """Get sliding window configuration."""
    return {
        'window_size': enhanced_config.model.sliding_window_size,
        'step_size': enhanced_config.model.sliding_window_step
    }

def validate_time_consistency():
    """Validate that all time configurations are consistent with the forecasting strategy."""
    base_minutes = enhanced_config.time.base_interval_minutes
    
    for horizon, steps in enhanced_config.time.forecast_horizons.items():
        if horizon in ['15min', '1h', '6h', '12h']:
            # These use 15-minute intervals
            total_minutes = steps * base_minutes
            expected_minutes = {
                '15min': 15, '1h': 60, '6h': 360, '12h': 720
            }
            if total_minutes != expected_minutes[horizon]:
                raise ValueError(f"{horizon} horizon should be {expected_minutes[horizon]} minutes, got {total_minutes}")
                
        elif horizon in ['24h', '48h']:
            # These use hourly intervals after upsampling
            total_hours = steps * 1  # 1 hour per step
            expected_hours = {
                '24h': 24, '48h': 48
            }
            if total_hours != expected_hours[horizon]:
                raise ValueError(f"{horizon} horizon should be {expected_hours[horizon]} hours, got {total_hours}")
    
    print("✅ Time configuration validation passed")
    return True

if __name__ == "__main__":
    # Test configuration
    validate_time_consistency()
    print("📊 Enhanced configuration loaded successfully")
    print(f"🕐 Base interval: {enhanced_config.time.base_interval_minutes} minutes")
    print(f"🎯 Forecast horizons: {list(enhanced_config.time.forecast_horizons.keys())}")
    print(f"📈 Direct 15-min horizons: {enhanced_config.model.direct_15min_horizons}")
    print(f"🔄 Direct hourly horizons: {enhanced_config.model.direct_hourly_horizons}")
    print(f"🎯 Upsampling horizons: {enhanced_config.time.upsampling_horizons}")
    print(f"📏 Sliding window: {enhanced_config.model.sliding_window_size} steps")